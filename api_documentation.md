# API Documentation

This document provides a comprehensive guide to integrating with the Player Search API.

## Endpoints

## Base URL

*   **Base URL:** `http://167.86.115.58:5000`

### 1. Health Check

*   **Endpoint:** `/health`
*   **Method:** `GET`
*   **Description:** Checks the health of the application and its services, including the database and the `pgvector` extension.
*   **Request Parameters:** None
*   **Example Request:**
    ```bash
    curl {base_url}/health
    ### 2. Search Players

*   **Endpoint:** `{base_url}/api/v1/search`
*   **Method:** `POST`
*   **Description:** Searches for players based on a variety of filters. This is the primary endpoint for finding players.
*   **Request Body:** A JSON object with the following parameters:
    *   `user_id` (integer, optional): The ID of the user performing the search. This is used for personalization and logging.
    *   `position` (string, optional): The player's position. Must be one of `forward`, `midfielder`, `defender`, `goalkeeper`, or `any`.
    *   `min_skill` (integer, optional): The minimum skill level of the player (1-100).
    *   `max_skill` (integer, optional): The maximum skill level of the player (1-100).
    *   `min_age` (integer, optional): The minimum age of the player (13-100).
    *   `max_age` (integer, optional): The maximum age of the player (13-100).
    *   `latitude` (float, optional): The latitude for location-based searches.
    *   `longitude` (float, optional): The longitude for location-based searches.
    *   `max_distance_km` (float, optional): The maximum distance in kilometers for location-based searches.
    *   `availability` (array of strings, optional): A list of availability tags (e.g., `weekday_evening`).
    *   `tags` (array of strings, optional): A list of player tags (e.g., `competitive`, `team-player`).
    *   `seed_player_ids` (array of strings, optional): A list of player IDs to use as a seed for finding similar players.
    *   `tag_boosts` (object, optional): A dictionary of tags and their boost values (e.g., `{"competitive": 1.5}`).
    *   `limit` (integer, optional): The maximum number of results to return (default: 20, max: 50).
    *   `offset` (integer, optional): The number of results to skip for pagination (default: 0).
*   **Example Request:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{
      "position": "midfielder",
      "min_skill": 70,
      "max_skill": 90,
      "limit": 10
    }' {base_url}/api/v1/search

    ### 3. Log Engagement Event

*   **Endpoint:** `{base_url}/api/v1/events`
*   **Method:** `POST`
*   **Description:** Logs user engagement events, such as impressions, profile views, and follows. This data is used to train the ML reranking model.
*   **Request Body:** A JSON object with the following parameters:
    *   `user_id` (string, required): The ID of the user who performed the event.
    *   `player_id` (string, required): The ID of the player the event is associated with.
    *   `event_type` (string, required): The type of event. Must be one of `impression`, `profile_view`, `follow`, `message`, or `save_to_playlist`.
    *   `query_context` (object, optional): A dictionary containing the context of the search query that led to this event.
*   **Example Request:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{
      "user_id": "some_user_id",
      "player_id": "some_player_id",
      "event_type": "profile_view"
    }' {base_url}/api/v1/events
    ### 4. Get Player Recommendations

*   **Endpoint:** `{base_url}/api/v1/recommendations/<string:player_id>`
*   **Method:** `GET`
*   **Description:** Gets a list of recommended players for a given player.
*   **URL Parameters:**
    *   `player_id` (string, required): The ID of the player to get recommendations for.
*   **Example Request:**
    ```bash
    curl {base_url}/api/v1/recommendations/some_player_id
    ### 5. Train ML Model

*   **Endpoint:** `{base_url}/api/v1/admin/train-model`
*   **Method:** `POST`
*   **Description:** Triggers the training of the ML reranking model.
*   **Example Request:**
    ```bash
    curl -X POST {base_url}/api/v1/admin/train-model
    ### 6. Saved Searches

*   **Endpoint:** `/api/v1/saved-searches`
*   **Method:** `POST`
*   **Description:** Creates a new saved search for a user.
*   **Request Body:** A JSON object with the following parameters:
    *   `user_id` (string, required): The ID of the user who is saving the search.
    *   `search_criteria` (object, required): A dictionary containing the search criteria.
*   **Example Request:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{
      "user_id": "some_user_id",
      "search_criteria": {
        "position": "Midfielder",
        "min_skill": 4
      }
    }' http://localhost:5000/api/v1/saved-searches

    ### 7. Generate Embeddings

*   **Endpoint:** `/api/v1/admin/generate-embeddings`
*   **Method:** `POST`
*   **Description:** Triggers the generation of embeddings for 100 players at one time. This is an admin endpoint and should be protected.
*   **Example Request:**
    ```bash
    curl -X POST {base_url}/api/v1/admin/generate-embeddings
    ```

*   **Endpoint:** `/api/v1/saved-searches/<int:search_id>/matches`
*   **Method:** `GET`
*   **Description:** Gets all matches for a saved search.
*   **URL Parameters:**
    *   `search_id` (int, required): The ID of the saved search.
*   **Example Request:**
    ```bash
    curl {base_url}/api/v1/saved-searches/{search_id}/matches
    ```
*   **Example Response:**
    ```json
    [
      {
        "id": "player_id_1",
        "name": "Player 1",
        "position": "Midfielder",
        "skill_level": 5,
        "age": 23,
        "location": "London, UK",
        "availability": "available"
      },
      {
        "id": "player_id_2",
        "name": "Player 2",
        "position": "Midfielder",
        "skill_level": 4,
        "age": 25,
        "location": "Manchester, UK",
        "availability": "available"
      }
    ]
    ```

*   **Endpoint:** `/api/v1/saved-searches/<user_id>`
*   **Method:** `GET`
*   **Description:** Gets a list of saved searches for a given user.
*   **URL Parameters:**
    *   `user_id` (string, required): The ID of the user to get saved searches for.
*   **Example Request:**
    ```bash
    curl {base_url}/api/v1/saved-searches/some_user_id
    ```
*   **Example Response:**
    ```json
    [
      {
        "id": 1,
        "user_id": "some_user_id",
        "search_criteria": {
          "position": "Midfielder",
          "min_skill": 4
        }
      }
    ]
    

# Post Recommendation System - Integration Guide

## Overview

I've added a complete post recommendation system to your application that includes:

1. **Post Interaction Tracking** - Log user interactions (views, likes, comments, shares, saves)
2. **ML-Based Personalization** - Train models on user behavior to rank posts
3. **Personalized Feed** - Get customized post feeds for each user
4. **Trending Posts** - Discover popular content with engagement-based ranking

---

## 🗄️ Step 1: Database Setup

Run the new schema to create the posts tables:

```bash
psql -h 167.86.115.58 -p 5432 -U devuser -d prospects_dev -f posts_schema.sql
```

This creates:
- `posts` - Store user-generated content
- `post_interactions` - Track all engagement events
- `post_comments` - Detailed comment data
- `post_engagement_stats` - Cached metrics for performance

---

## 📦 Step 2: Install Dependencies

The PostRecommender uses the same dependencies as your existing ML pipeline:

```bash
pip install lightgbm scikit-learn numpy
```

---

## 🚀 Step 3: API Endpoints

### Log Post Interactions

Track when users interact with posts:

```bash
POST /api/v1/posts/interactions
```

**Request Body:**
```json
{
  "user_id": "01HXA7B2C3D4E5F6G7H8J9K0M1",
  "post_id": "01HXA7B2C3D4E5F6G7H8J9K0M2",
  "interaction_type": "like",
  "interaction_metadata": {
    "source": "feed",
    "device": "mobile"
  },
  "dwell_time_seconds": 5.2
}
```

**Interaction Types:**
- `view` - User viewed the post (weight: 0)
- `like` - User liked the post (weight: 2)
- `comment` - User commented (weight: 4)
- `share` - User shared (weight: 5)
- `save` - User saved for later (weight: 3)

**Response:**
```json
{
  "status": "logged",
  "interaction_id": 12345,
  "interaction_type": "like",
  "timestamp": "2025-12-11T10:30:00Z"
}
```

---

### Get Personalized Feed

Retrieve a customized feed for a specific user:

```bash
GET /api/v1/posts/feed/{user_id}?limit=20&offset=0
```

**Example:**
```bash
curl -X GET "http://localhost:5000/api/v1/posts/feed/01HXA7B2C3D4E5F6G7H8J9K0M1?limit=20&offset=0"
```

**Response:**
```json
{
  "posts": [
    {
      "id": "01HXA7B2C3D4E5F6G7H8J9K0M2",
      "player_id": "01HXA7B2C3D4E5F6G7H8J9K0M3",
      "content": "Great training session today! 💪",
      "media_urls": ["https://cdn.example.com/image1.jpg"],
      "hashtags": ["training", "fitness"],
      "created_at": "2025-12-11T09:00:00Z",
      "author": {
        "name": "John Doe",
        "avatar": "https://cdn.example.com/avatar.jpg"
      },
      "engagement": {
        "likes": 45,
        "comments": 12,
        "shares": 3
      },
      "recommendation_score": 0.87,
      "from_followed_author": true
    }
  ],
  "total": 150,
  "user_id": "01HXA7B2C3D4E5F6G7H8J9K0M1",
  "metadata": {
    "personalized": true,
    "recommendation_strategy": "ml_ranking",
    "cached": false
  },
  "pagination": {
    "limit": 20,
    "offset": 0,
    "has_more": true
  }
}
```

---

### Get Trending Posts

Fetch posts that are currently popular:

```bash
GET /api/v1/posts/trending?limit=20&timeframe=24h
```

**Timeframes:**
- `1h`, `6h`, `12h`, `24h` (default), `48h`, `7d`

**Response:**
```json
{
  "posts": [
    {
      "id": "01HXA7B2C3D4E5F6G7H8J9K0M2",
      "player_id": "01HXA7B2C3D4E5F6G7H8J9K0M3",
      "content": "Just scored the winning goal! 🎯⚽",
      "media_urls": ["https://cdn.example.com/goal.mp4"],
      "created_at": "2025-12-11T08:30:00Z",
      "author": {
        "name": "Sarah Smith",
        "avatar": "https://cdn.example.com/sarah.jpg"
      },
      "engagement": {
        "likes": 234,
        "comments": 67,
        "shares": 45,
        "unique_users": 189
      },
      "engagement_score": 156.7
    }
  ],
  "total": 20,
  "metadata": {
    "timeframe": "24h",
    "hours": 24,
    "type": "trending"
  }
}
```

---

## 🤖 Step 4: Train the ML Model

### First Time Setup

Train the initial model:

```bash
# Via API
curl -X POST http://localhost:5000/api/v1/admin/train-post-model

# Via command line
python app.py train-post-model
```

**Response:**
```json
{
  "message": "Model trained and saved successfully",
  "metadata": {
    "samples": 15420,
    "user_groups": 347,
    "model_path": "models/post_recommender.pkl"
  }
}
```

### Scheduled Retraining

Set up a cron job or scheduled task to retrain periodically:

```bash
# Retrain daily at 3 AM
0 3 * * * cd /path/to/app && python app.py train-post-model >> logs/training.log 2>&1
```

Or use a task scheduler like Celery:

```python
from celery import Celery
from celery.schedules import crontab

@celery.task
def train_post_model():
    result = post_recommender.train_model()
    return result

# Schedule
celery.conf.beat_schedule = {
    'train-post-model-daily': {
        'task': 'tasks.train_post_model',
        'schedule': crontab(hour=3, minute=0),
    },
}
```

---

## 🧪 Step 5: Testing the System

### Create Test Posts

```sql
-- Insert sample posts
INSERT INTO posts (id, player_id, content, hashtags, created_at) VALUES
('01HXA7B2C3D4E5F6G7H8J9K0M2', '01HXA7B2C3D4E5F6G7H8J9K0M3', 'Training hard for the big game!', ARRAY['training', 'preparation'], NOW() - INTERVAL '2 hours'),
('01HXA7B2C3D4E5F6G7H8J9K0M4', '01HXA7B2C3D4E5F6G7H8J9K0M5', 'New personal best on sprint time! 🏃‍♂️', ARRAY['fitness', 'achievement'], NOW() - INTERVAL '5 hours'),
('01HXA7B2C3D4E5F6G7H8J9K0M6', '01HXA7B2C3D4E5F6G7H8J9K0M7', 'Team practice highlights 📹', ARRAY['teamwork', 'practice'], NOW() - INTERVAL '1 day');
```

### Simulate User Interactions

```bash
# User views a post
curl -X POST http://localhost:5000/api/v1/posts/interactions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "01HXA7B2C3D4E5F6G7H8J9K0M1",
    "post_id": "01HXA7B2C3D4E5F6G7H8J9K0M2",
    "interaction_type": "view",
    "dwell_time_seconds": 3.5
  }'

# User likes the post
curl -X POST http://localhost:5000/api/v1/posts/interactions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "01HXA7B2C3D4E5F6G7H8J9K0M1",
    "post_id": "01HXA7B2C3D4E5F6G7H8J9K0M2",
    "interaction_type": "like"
  }'
```

### Test Personalized Feed

```bash
# Get feed (will use fallback ranking before model is trained)
curl -X GET "http://localhost:5000/api/v1/posts/feed/01HXA7B2C3D4E5F6G7H8J9K0M1?limit=10"
```

---

## 📊 How the ML Model Works

### Features Used for Ranking

The model considers 14 features per post:

1. **User-Post Affinity** - How often user engages with this author
2. **Post Age** - Recency of the post
3. **Author Popularity** - Follower count
4. **Engagement Metrics** - Likes, comments, shares
5. **User Behavior** - Like rate, comment rate, share rate
6. **Similarity** - Author-user similarity
7. **Timing** - Time-of-day match
8. **Content Type** - Video/image/text preference
9. **Hashtag Overlap** - Interest alignment
10. **Engagement Velocity** - How fast post is gaining traction

### Training Strategy

- **Algorithm**: LightGBM LambdaRank (Learning to Rank)
- **Objective**: Maximize NDCG@5,10,20
- **Groups**: Per-user (each user's interaction history is a group)
- **Labels**: Engagement scores (0=view, 2=like, 4=comment, 5=share)

### Ranking Logic

Posts are scored based on predicted engagement probability. Higher scores appear first in the feed.

---

## 🎯 Frontend Integration

### Example React Component

```javascript
import React, { useState, useEffect } from 'react';

function PersonalizedFeed({ userId }) {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchFeed();
  }, [userId]);

  const fetchFeed = async () => {
    try {
      const response = await fetch(
        `http://localhost:5000/api/v1/posts/feed/${userId}?limit=20`
      );
      const data = await response.json();
      setPosts(data.posts);
    } catch (error) {
      console.error('Failed to fetch feed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLike = async (postId) => {
    await fetch('http://localhost:5000/api/v1/posts/interactions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        post_id: postId,
        interaction_type: 'like'
      })
    });
    // Refresh feed
    fetchFeed();
  };

  if (loading) return <div>Loading feed...</div>;

  return (
    <div className="feed">
      {posts.map(post => (
        <div key={post.id} className="post-card">
          <div className="post-header">
            <img src={post.author.avatar} alt={post.author.name} />
            <span>{post.author.name}</span>
          </div>
          <div className="post-content">{post.content}</div>
          {post.media_urls.map(url => (
            <img key={url} src={url} alt="Post media" />
          ))}
          <div className="post-actions">
            <button onClick={() => handleLike(post.id)}>
              ❤️ {post.engagement.likes}
            </button>
            <button>💬 {post.engagement.comments}</button>
            <button>🔄 {post.engagement.shares}</button>
          </div>
        </div>
      ))}
    </div>
  );
}
```

---

## 📈 Monitoring & Analytics

### Check Model Performance

```sql
-- Average engagement by recommendation score quartile
SELECT 
    NTILE(4) OVER (ORDER BY recommendation_score) as quartile,
    AVG(like_count) as avg_likes,
    AVG(comment_count) as avg_comments,
    AVG(share_count) as avg_shares
FROM personalized_feed_logs
GROUP BY quartile
ORDER BY quartile;
```

### Track User Engagement

```sql
-- User engagement trends
SELECT 
    DATE(created_at) as date,
    interaction_type,
    COUNT(*) as interaction_count
FROM post_interactions
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY date, interaction_type
ORDER BY date DESC;
```

---

## 🔧 Configuration

Add to your `.env` file:

```bash
# Post recommendation settings
POST_MODEL_PATH=models/post_recommender.pkl
POST_FEED_CACHE_TTL=300  # 5 minutes
MIN_TRAINING_SAMPLES=100
RETRAIN_FREQUENCY_HOURS=24
```

    