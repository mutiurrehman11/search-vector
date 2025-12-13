"""
Post Recommendation Engine - Personalized content feed using ML
"""

import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PostRecommendationEngine:
    """LightGBM-based recommendation engine for personalized post feeds"""

    def __init__(self, db_conn, model_path: str = "models/post_recommender.pkl"):
        self.conn = db_conn
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path

        self.feature_names = [
            'user_post_affinity',  # How often user interacts with this author
            'post_age_hours',  # How old the post is
            'author_follower_count',  # Author's popularity
            'post_like_count',  # Total likes on post
            'post_comment_count',  # Total comments
            'post_share_count',  # Total shares
            'user_like_rate',  # User's overall like rate
            'user_comment_rate',  # User's overall comment rate
            'user_share_rate',  # User's overall share rate
            'author_user_similarity',  # Similarity between author and user
            'time_of_day_match',  # Does posting time match user's active hours
            'content_type_preference',  # User preference for content type (video/image/text)
            'hashtag_overlap',  # Overlap with user's interests
            'engagement_velocity'  # How fast the post is gaining engagement
        ]

        # Load existing model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info(f"Loaded post recommendation model from {model_path}")
        else:
            # Fit scaler with dummy data
            self.scaler.fit(np.zeros((1, len(self.feature_names))))

    def extract_features(self, user_id: str, post: Dict, user_stats: Dict) -> List[float]:
        """Extract features for a user-post pair"""
        features = []

        # User-post affinity (how often user interacts with this author)
        user_post_affinity = user_stats.get('author_affinity', {}).get(post['player_id'], 0.0)
        features.append(user_post_affinity)

        # Post age in hours
        post_age = (datetime.now() - post['created_at']).total_seconds() / 3600.0
        features.append(post_age)

        # Author metrics
        features.append(post.get('author_follower_count', 0))

        # Post engagement metrics
        features.append(post.get('like_count', 0))
        features.append(post.get('comment_count', 0))
        features.append(post.get('share_count', 0))

        # User engagement rates
        features.append(user_stats.get('like_rate', 0.1))
        features.append(user_stats.get('comment_rate', 0.05))
        features.append(user_stats.get('share_rate', 0.02))

        # Author-user similarity (based on positions, skills, etc.)
        features.append(post.get('author_similarity', 0.5))

        # Time of day match (placeholder - would need user's active hours)
        post_hour = post['created_at'].hour
        user_active_hours = user_stats.get('active_hours', [8, 12, 18, 22])
        time_match = 1.0 if post_hour in user_active_hours else 0.3
        features.append(time_match)

        # Content type preference
        content_type = post.get('content_type', 'text')
        user_pref = user_stats.get('content_preferences', {}).get(content_type, 0.5)
        features.append(user_pref)

        # Hashtag overlap
        post_tags = set(post.get('hashtags', []))
        user_tags = set(user_stats.get('interest_tags', []))
        hashtag_overlap = len(post_tags & user_tags) / max(len(post_tags | user_tags), 1)
        features.append(hashtag_overlap)

        # Engagement velocity (engagement per hour since posting)
        total_engagement = post.get('like_count', 0) + post.get('comment_count', 0) * 2 + post.get('share_count', 0) * 3
        velocity = total_engagement / max(post_age, 0.1)
        features.append(velocity)

        return features

    def get_user_statistics(self, user_id: str) -> Dict:
        """Get user engagement statistics and preferences"""
        try:
            with self.conn.cursor() as cur:
                # Get user's interaction rates
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_interactions,
                        COUNT(*) FILTER (WHERE interaction_type = 'like') as likes,
                        COUNT(*) FILTER (WHERE interaction_type = 'comment') as comments,
                        COUNT(*) FILTER (WHERE interaction_type = 'share') as shares,
                        COUNT(*) FILTER (WHERE interaction_type = 'view') as views
                    FROM post_interactions
                    WHERE user_id = %s
                    AND created_at > NOW() - INTERVAL '30 days';
                """, (user_id,))

                row = cur.fetchone()
                if not row or row[0] == 0:
                    # New user - return defaults
                    return {
                        'like_rate': 0.1,
                        'comment_rate': 0.05,
                        'share_rate': 0.02,
                        'author_affinity': {},
                        'interest_tags': [],
                        'content_preferences': {'text': 0.5, 'image': 0.5, 'video': 0.5},
                        'active_hours': [8, 12, 18, 22]
                    }

                total_interactions = row[0]
                views = row[4] if row[4] > 0 else total_interactions

                # Calculate rates
                like_rate = row[1] / views if views > 0 else 0.1
                comment_rate = row[2] / views if views > 0 else 0.05
                share_rate = row[3] / views if views > 0 else 0.02

                # Get author affinity (authors user interacts with most)
                cur.execute("""
                    SELECT 
                        p.player_id,
                        COUNT(*) as interaction_count,
                        COUNT(*) / %s::float as affinity_score
                    FROM post_interactions pi
                    JOIN posts p ON pi.post_id = p.id
                    WHERE pi.user_id = %s
                    AND pi.created_at > NOW() - INTERVAL '30 days'
                    AND pi.interaction_type IN ('like', 'comment', 'share')
                    GROUP BY p.player_id
                    ORDER BY interaction_count DESC
                    LIMIT 50;
                """, (max(total_interactions, 1), user_id))

                author_affinity = {row[0]: row[2] for row in cur.fetchall()}

                # Get user's interest tags (from posts they engaged with)
                cur.execute("""
                    SELECT 
                        UNNEST(p.hashtags) as tag,
                        COUNT(*) as tag_count
                    FROM post_interactions pi
                    JOIN posts p ON pi.post_id = p.id
                    WHERE pi.user_id = %s
                    AND pi.created_at > NOW() - INTERVAL '30 days'
                    AND pi.interaction_type IN ('like', 'comment', 'share', 'save')
                    AND p.hashtags IS NOT NULL
                    GROUP BY tag
                    ORDER BY tag_count DESC
                    LIMIT 20;
                """, (user_id,))

                interest_tags = [row[0] for row in cur.fetchall()]

                return {
                    'like_rate': like_rate,
                    'comment_rate': comment_rate,
                    'share_rate': share_rate,
                    'author_affinity': author_affinity,
                    'interest_tags': interest_tags,
                    'content_preferences': {'text': 0.5, 'image': 0.5, 'video': 0.5},
                    'active_hours': [8, 12, 18, 22]
                }

        except Exception as e:
            logger.error(f"Error getting user statistics: {e}")
            return {
                'like_rate': 0.1,
                'comment_rate': 0.05,
                'share_rate': 0.02,
                'author_affinity': {},
                'interest_tags': [],
                'content_preferences': {'text': 0.5, 'image': 0.5, 'video': 0.5},
                'active_hours': [8, 12, 18, 22]
            }

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from post interactions"""
        try:
            with self.conn.cursor() as cur:
                # Get interactions with post and author data
                cur.execute("""
                    SELECT 
                        pi.user_id,
                        pi.post_id,
                        pi.interaction_type,
                        pi.dwell_time_seconds,
                        p.player_id as author_id,
                        p.content,
                        p.hashtags,
                        p.created_at,
                        EXTRACT(EPOCH FROM (NOW() - p.created_at)) / 3600.0 as post_age_hours,
                        COUNT(*) FILTER (WHERE pi2.interaction_type = 'like') as like_count,
                        COUNT(*) FILTER (WHERE pi2.interaction_type = 'comment') as comment_count,
                        COUNT(*) FILTER (WHERE pi2.interaction_type = 'share') as share_count,
                        (SELECT COUNT(*) FROM follows WHERE followedable_id = p.player_id) as author_follower_count
                    FROM post_interactions pi
                    JOIN posts p ON pi.post_id = p.id
                    LEFT JOIN post_interactions pi2 ON p.id = pi2.post_id
                    WHERE pi.created_at > NOW() - INTERVAL '60 days'
                    AND p.deleted_at IS NULL
                    GROUP BY pi.user_id, pi.post_id, pi.interaction_type, pi.dwell_time_seconds,
                             p.player_id, p.content, p.hashtags, p.created_at
                    ORDER BY pi.user_id, pi.created_at;
                """)

                rows = cur.fetchall()

                if not rows:
                    logger.warning("No training data available for post recommendations")
                    return np.array([]), np.array([]), np.array([])

                X, y, group_ids = [], [], []
                current_user = None
                current_group_id = -1
                user_stats_cache = {}

                for row in rows:
                    user_id = row[0]

                    # New user = new group
                    if user_id != current_user:
                        current_group_id += 1
                        current_user = user_id

                        # Cache user stats
                        if user_id not in user_stats_cache:
                            user_stats_cache[user_id] = self.get_user_statistics(user_id)

                    # Build post dict
                    post = {
                        'post_id': row[1],
                        'player_id': row[4],
                        'content': row[5],
                        'hashtags': row[6] if row[6] else [],
                        'created_at': row[7],
                        'post_age_hours': row[8],
                        'like_count': row[9],
                        'comment_count': row[10],
                        'share_count': row[11],
                        'author_follower_count': row[12],
                        'author_similarity': 0.5,  # Would compute from embeddings
                        'content_type': 'text'  # Would parse from content
                    }

                    # Extract features
                    features = self.extract_features(user_id, post, user_stats_cache[user_id])
                    X.append(features)

                    # Create engagement score (target)
                    interaction_type = row[2]
                    dwell_time = row[3]

                    engagement_scores = {
                        'view': 0,
                        'like': 2,
                        'comment': 4,
                        'share': 5,
                        'save': 3
                    }

                    base_score = engagement_scores.get(interaction_type, 0)
                    # Boost score if user spent time on post
                    if dwell_time > 10:
                        base_score += 1

                    y.append(base_score)
                    group_ids.append(current_group_id)

                # Convert to arrays
                X = np.array(X)
                y = np.array(y)
                group_ids = np.array(group_ids)

                # Get group counts
                unique_groups, group_counts = np.unique(group_ids, return_counts=True)

                logger.info(f"Training data prepared: {len(X)} samples, {len(unique_groups)} user groups")
                logger.info(
                    f"Group sizes - min: {group_counts.min()}, max: {group_counts.max()}, mean: {group_counts.mean():.1f}")

                return X, y, group_counts

        except Exception as e:
            logger.error(f"Error preparing training data: {e}", exc_info=True)
            return np.array([]), np.array([]), np.array([])

    def train_model(self) -> Dict:
        """Train the post recommendation model"""
        logger.info("Starting post recommendation model training...")

        X, y, group_counts = self.prepare_training_data()

        if len(X) == 0:
            logger.warning("No training data available")
            return {
                'success': False,
                'error': 'No training data available',
                'metadata': {'samples': 0}
            }

        if len(group_counts) < 2:
            logger.warning(f"Insufficient user groups: {len(group_counts)}")
            return {
                'success': False,
                'error': 'Need at least 2 users with interaction history',
                'metadata': {'groups': len(group_counts)}
            }

        try:
            # Standardize features
            X_scaled = self.scaler.fit_transform(X)

            # Create LightGBM dataset
            train_data = lgb.Dataset(
                X_scaled,
                label=y,
                group=group_counts,
                feature_name=self.feature_names
            )

            # LambdaRank parameters
            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [5, 10, 20],
                'learning_rate': 0.05,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'force_row_wise': True,
                'lambdarank_norm': True
            }

            logger.info(f"Training post recommendation model with {len(group_counts)} user groups")

            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(10, verbose=False)]
            )

            # Log feature importance
            importance = self.model.feature_importance(importance_type='gain')
            for fname, imp in zip(self.feature_names, importance):
                logger.info(f"Feature importance - {fname}: {imp:.4f}")

            # Save model
            self.save_model(self.model_path)

            logger.info("Post recommendation model trained successfully")
            return {
                'success': True,
                'message': 'Model trained and saved successfully',
                'metadata': {
                    'samples': len(X),
                    'user_groups': len(group_counts),
                    'model_path': self.model_path
                }
            }

        except Exception as e:
            logger.error(f"Error training post model: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'metadata': {'samples': len(X)}
            }

    def get_personalized_feed(self, user_id: str, limit: int = 20, offset: int = 0) -> Dict:
        """Get personalized post feed for a user"""
        try:
            # Get user statistics
            user_stats = self.get_user_statistics(user_id)

            # Get candidate posts (recent posts from followed authors + trending)
            with self.conn.cursor() as cur:
                cur.execute("""
                    WITH user_follows AS (
                        SELECT followedable_id as followed_player_id
                        FROM follows
                        WHERE followerable_id = %s
                        AND followerable_type = 'Player'
                    ),
                    candidate_posts AS (
                        SELECT DISTINCT
                            p.id,
                            p.player_id,
                            p.content,
                            p.media_urls,
                            p.hashtags,
                            p.created_at,
                            CONCAT(pl.first_name, ' ', pl.last_name) as author_name,
                            pl.profile_picture as author_avatar,
                            COUNT(DISTINCT CASE WHEN pi.interaction_type = 'like' THEN pi.id END) as like_count,
                            COUNT(DISTINCT CASE WHEN pi.interaction_type = 'comment' THEN pi.id END) as comment_count,
                            COUNT(DISTINCT CASE WHEN pi.interaction_type = 'share' THEN pi.id END) as share_count,
                            (SELECT COUNT(*) FROM follows WHERE followedable_id = p.player_id) as author_follower_count,
                            CASE 
                                WHEN uf.followed_player_id IS NOT NULL THEN 1 
                                ELSE 0 
                            END as from_followed_author
                        FROM posts p
                        JOIN players pl ON p.player_id = pl.id
                        LEFT JOIN user_follows uf ON p.player_id = uf.followed_player_id
                        LEFT JOIN post_interactions pi ON p.id = pi.post_id
                        WHERE p.created_at > NOW() - INTERVAL '7 days'
                        AND p.deleted_at IS NULL
                        AND pl.status = 'active'
                        AND NOT EXISTS (
                            SELECT 1 FROM post_interactions pi2
                            WHERE pi2.post_id = p.id
                            AND pi2.user_id = %s
                            AND pi2.interaction_type = 'view'
                        )
                        GROUP BY p.id, pl.first_name, pl.last_name, pl.profile_picture, uf.followed_player_id
                        ORDER BY from_followed_author DESC, p.created_at DESC
                        LIMIT 200
                    )
                    SELECT * FROM candidate_posts;
                """, (user_id, user_id))

                candidate_rows = cur.fetchall()

            if not candidate_rows:
                logger.info(f"No candidate posts found for user {user_id}")
                return {
                    'posts': [],
                    'total_count': 0,
                    'personalized': False,
                    'strategy': 'no_posts_available'
                }

            # If model is not trained, return posts sorted by recency and popularity
            if not self.model or not hasattr(self.scaler, 'mean_'):
                logger.info("Model not trained - using fallback ranking")
                posts = self._format_posts(candidate_rows)
                return {
                    'posts': posts[offset:offset + limit],
                    'total_count': len(posts),
                    'personalized': False,
                    'strategy': 'fallback_chronological'
                }

            # Extract features and score each post
            scored_posts = []
            for row in candidate_rows:
                post = {
                    'id': row[0],
                    'player_id': row[1],
                    'content': row[2],
                    'media_urls': row[3] if row[3] else [],
                    'hashtags': row[4] if row[4] else [],
                    'created_at': row[5],
                    'author_name': row[6],
                    'author_avatar': row[7],
                    'like_count': row[8],
                    'comment_count': row[9],
                    'share_count': row[10],
                    'author_follower_count': row[11],
                    'from_followed_author': row[12],
                    'author_similarity': 0.5,
                    'content_type': 'text'
                }

                features = self.extract_features(user_id, post, user_stats)
                scored_posts.append((post, features))

            # Score using ML model
            X = np.array([features for _, features in scored_posts])
            X_scaled = self.scaler.transform(X)
            scores = self.model.predict(X_scaled)

            # Combine posts with scores
            for (post, _), score in zip(scored_posts, scores):
                post['recommendation_score'] = float(score)

            # Sort by score
            ranked_posts = sorted(
                [post for post, _ in scored_posts],
                key=lambda p: p['recommendation_score'],
                reverse=True
            )

            # Format and paginate
            formatted_posts = []
            for post in ranked_posts[offset:offset + limit]:
                formatted_posts.append({
                    'id': post['id'],
                    'player_id': post['player_id'],
                    'content': post['content'],
                    'media_urls': post['media_urls'],
                    'hashtags': post['hashtags'],
                    'created_at': post['created_at'].isoformat(),
                    'author': {
                        'name': post['author_name'],
                        'avatar': post['author_avatar']
                    },
                    'engagement': {
                        'likes': post['like_count'],
                        'comments': post['comment_count'],
                        'shares': post['share_count']
                    },
                    'recommendation_score': post['recommendation_score'],
                    'from_followed_author': bool(post['from_followed_author'])
                })

            logger.info(f"Generated personalized feed for user {user_id}: {len(formatted_posts)} posts")

            return {
                'posts': formatted_posts,
                'total_count': len(ranked_posts),
                'personalized': True,
                'strategy': 'ml_ranking'
            }

        except Exception as e:
            logger.error(f"Error generating personalized feed: {e}", exc_info=True)
            return {
                'posts': [],
                'total_count': 0,
                'personalized': False,
                'strategy': 'error',
                'error': str(e)
            }

    def _format_posts(self, rows: List) -> List[Dict]:
        """Format post rows into dicts"""
        posts = []
        for row in rows:
            posts.append({
                'id': row[0],
                'player_id': row[1],
                'content': row[2],
                'media_urls': row[3] if row[3] else [],
                'hashtags': row[4] if row[4] else [],
                'created_at': row[5].isoformat(),
                'author': {
                    'name': row[6],
                    'avatar': row[7]
                },
                'engagement': {
                    'likes': row[8],
                    'comments': row[9],
                    'shares': row[10]
                }
            })
        return posts

    def save_model(self, path: str):
        """Save trained model and scaler"""
        if not self.model:
            logger.warning("No model to save")
            return

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }

            with open(path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Post recommendation model saved to {path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)

    def load_model(self, path: str) -> bool:
        """Load trained model and scaler"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data.get('feature_names', self.feature_names)

            logger.info(f"Post recommendation model loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return False