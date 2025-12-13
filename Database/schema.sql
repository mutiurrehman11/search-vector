-- extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "btree_gist";
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS cube;
CREATE EXTENSION IF NOT EXISTS earthdistance;

-- Tenants table - for multi-tenancy
CREATE TABLE tenants (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    url JSONB,
    active BOOLEAN DEFAULT TRUE,
    sport_type VARCHAR(255),
    sport_config JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Players table – core profile with embedding column integrated
CREATE TABLE players (
    id CHAR(26) PRIMARY KEY,
    first_name VARCHAR NOT NULL,
    last_name VARCHAR NOT NULL,
    contacts JSONB,
    description TEXT,
    playing_since DATE,
    height REAL,
    weight REAL,
    location JSONB NOT NULL,
    birth_date DATE NOT NULL,
    tags JSONB,
    availability JSONB,
    document VARCHAR,
    profile_picture JSONB,
    pictures JSONB,
    gender VARCHAR,
    status VARCHAR NOT NULL,
    user_id CHAR(26),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    tenant_id VARCHAR(255) REFERENCES tenants(id),
    position_code VARCHAR(10),
    embedding vector(128)  -- Integrated embedding column
);

-- Player skills
CREATE TABLE player_skills (
    id BIGSERIAL PRIMARY KEY,
    player_id CHAR(26) NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    skill VARCHAR NOT NULL CHECK (skill IN ('dribbling','passing','shooting','speed','stamina','strength')),
    level INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Player-team association
CREATE TABLE player_teams (
    id BIGSERIAL PRIMARY KEY,
    player_id CHAR(26) NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    playable_type VARCHAR NOT NULL,
    playable_id CHAR(26) NOT NULL,
    overview TEXT,
    start_at TIMESTAMPTZ NOT NULL,
    end_at TIMESTAMPTZ,
    position VARCHAR,
    tshirt INTEGER,
    approved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Player improvements table - Performance tracking
CREATE TABLE player_improvements (
    id BIGSERIAL PRIMARY KEY,
    player_id CHAR(26) NOT NULL,
    assessment_date DATE NOT NULL,
    skill_score DECIMAL(5,2),
    physical_score DECIMAL(5,2),
    mental_score DECIMAL(5,2),
    overall_score DECIMAL(5,2),
    improvement_status VARCHAR(255),
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    FOREIGN KEY (player_id) REFERENCES players(id)
);

-- Player goals table - Goal tracking
CREATE TABLE player_goals (
    id BIGSERIAL PRIMARY KEY,
    player_id CHAR(26) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    target_date DATE,
    status VARCHAR(255) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    FOREIGN KEY (player_id) REFERENCES players(id)
);

-- Player training sessions table - Training data
CREATE TABLE player_training_sessions (
    id BIGSERIAL PRIMARY KEY,
    player_id CHAR(26) NOT NULL,
    session_date DATE NOT NULL,
    duration_minutes INT,
    intensity VARCHAR(255),
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    FOREIGN KEY (player_id) REFERENCES players(id)
);

-- Interaction tables
CREATE TABLE views (
    id BIGSERIAL PRIMARY KEY,
    ip VARCHAR NOT NULL,
    viewable_type VARCHAR NOT NULL,
    viewable_id CHAR(26) NOT NULL,
    viewerable_type VARCHAR NOT NULL,
    viewerable_id CHAR(26) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE follows (
    id BIGSERIAL PRIMARY KEY,
    followerable_type VARCHAR NOT NULL,
    followerable_id CHAR(26) NOT NULL,
    followedable_type VARCHAR NOT NULL,
    followedable_id CHAR(26) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chat_rooms (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    slug VARCHAR UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chat_messages (
    id BIGSERIAL PRIMARY KEY,
    chat_room_id BIGINT NOT NULL REFERENCES chat_rooms(id) ON DELETE CASCADE,
    sender_type VARCHAR NOT NULL,
    sender_id CHAR(26) NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE playlists (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    description TEXT,
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Skills reference table
CREATE TABLE skills (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Categories table - Player categories
CREATE TABLE categories (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Divisions table - Division data
CREATE TABLE divisions (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    level INT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Soccer positions table - Position reference
CREATE TABLE soccer_positions (
    id BIGSERIAL PRIMARY KEY,
    code VARCHAR(255) UNIQUE NOT NULL,
    name_en VARCHAR(255) NOT NULL,
    name_pt VARCHAR(255),
    category VARCHAR(255),
    tenant_id VARCHAR(255) REFERENCES tenants(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Geographic lookup tables
CREATE TABLE countries (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    code CHAR(3) UNIQUE NOT NULL
);

CREATE TABLE states (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    code VARCHAR,
    country_id BIGINT NOT NULL REFERENCES countries(id)
);

CREATE TABLE cities (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    state_id BIGINT NOT NULL REFERENCES states(id)
);

-- Player engagement stats
CREATE TABLE player_engagement_stats (
    player_id CHAR(26) PRIMARY KEY REFERENCES players(id) ON DELETE CASCADE,
    ctr DOUBLE PRECISION DEFAULT 0,
    recent_activity_score DOUBLE PRECISION DEFAULT 0,
    follow_rate DOUBLE PRECISION DEFAULT 0,
    message_rate DOUBLE PRECISION DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Player engagement events for ML training
CREATE TABLE player_engagement_events (
    id BIGSERIAL PRIMARY KEY,
    user_id CHAR(26) NOT NULL,
    player_id CHAR(26) NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    event_type VARCHAR NOT NULL CHECK (event_type IN ('impression','profile_view','follow','message','save_to_playlist')),
    query_context JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Saved searches
CREATE TABLE saved_searches (
    id BIGSERIAL PRIMARY KEY,
    user_id CHAR(26) NOT NULL,
    search_name VARCHAR NOT NULL,
    filters JSONB NOT NULL,
    query_embedding vector(128) NOT NULL,
    alert_frequency VARCHAR NOT NULL DEFAULT 'weekly',
    last_alerted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);


-- Posts table - User-generated content
CREATE TABLE posts (
    id CHAR(26) PRIMARY KEY,
    player_id CHAR(26) NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    media_urls JSONB,  -- Array of image/video URLs
    hashtags TEXT[],   -- Array of hashtags
    visibility VARCHAR(50) DEFAULT 'public' CHECK (visibility IN ('public', 'followers', 'private')),
    post_type VARCHAR(50) DEFAULT 'standard' CHECK (post_type IN ('standard', 'highlight', 'achievement', 'training')),
    location JSONB,    -- Optional location data
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    tenant_id VARCHAR(255) REFERENCES tenants(id)
);

-- Post interactions table - Track all user engagement with posts
CREATE TABLE post_interactions (
    id BIGSERIAL PRIMARY KEY,
    user_id CHAR(26) NOT NULL,  -- User who performed the interaction
    post_id CHAR(26) NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    interaction_type VARCHAR(50) NOT NULL CHECK (
        interaction_type IN ('view', 'like', 'comment', 'share', 'save')
    ),
    interaction_metadata JSONB,  -- Additional context (e.g., source: 'feed', 'profile', 'search')
    dwell_time_seconds REAL DEFAULT 0,  -- How long user viewed the post
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Comments table - Detailed comment data
CREATE TABLE post_comments (
    id CHAR(26) PRIMARY KEY,
    post_id CHAR(26) NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    player_id CHAR(26) NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    parent_comment_id CHAR(26) REFERENCES post_comments(id) ON DELETE CASCADE,  -- For threaded comments
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    deleted_at TIMESTAMPTZ
);

-- Post engagement aggregates - Cached counts for performance
CREATE TABLE post_engagement_stats (
    post_id CHAR(26) PRIMARY KEY REFERENCES posts(id) ON DELETE CASCADE,
    view_count INT DEFAULT 0,
    like_count INT DEFAULT 0,
    comment_count INT DEFAULT 0,
    share_count INT DEFAULT 0,
    save_count INT DEFAULT 0,
    unique_viewers INT DEFAULT 0,
    avg_dwell_time_seconds REAL DEFAULT 0,
    engagement_rate REAL DEFAULT 0,  -- (likes + comments + shares) / views
    viral_score REAL DEFAULT 0,      -- Weighted engagement with recency boost
    last_updated TIMESTAMPTZ DEFAULT now()
);

-- Indexes for posts
CREATE INDEX idx_posts_player_id ON posts(player_id);
CREATE INDEX idx_posts_created_at ON posts(created_at DESC);
CREATE INDEX idx_posts_visibility ON posts(visibility);
CREATE INDEX idx_posts_post_type ON posts(post_type);
CREATE INDEX idx_posts_deleted_at ON posts(deleted_at);
CREATE INDEX idx_posts_tenant_id ON posts(tenant_id);
CREATE INDEX idx_posts_hashtags ON posts USING gin(hashtags);  -- GIN index for array search

-- Indexes for post_interactions
CREATE INDEX idx_post_interactions_user_id ON post_interactions(user_id);
CREATE INDEX idx_post_interactions_post_id ON post_interactions(post_id);
CREATE INDEX idx_post_interactions_type ON post_interactions(interaction_type);
CREATE INDEX idx_post_interactions_created_at ON post_interactions(created_at DESC);
CREATE INDEX idx_post_interactions_user_post ON post_interactions(user_id, post_id);

-- Indexes for post_comments
CREATE INDEX idx_post_comments_post_id ON post_comments(post_id);
CREATE INDEX idx_post_comments_player_id ON post_comments(player_id);
CREATE INDEX idx_post_comments_parent_id ON post_comments(parent_comment_id);
CREATE INDEX idx_post_comments_created_at ON post_comments(created_at DESC);
CREATE INDEX idx_post_comments_deleted_at ON post_comments(deleted_at);

-- Function to update post engagement stats (trigger-based)
CREATE OR REPLACE FUNCTION update_post_engagement_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update or insert engagement stats
    INSERT INTO post_engagement_stats (
        post_id,
        view_count,
        like_count,
        comment_count,
        share_count,
        save_count,
        unique_viewers,
        last_updated
    )
    SELECT
        NEW.post_id,
        COUNT(*) FILTER (WHERE interaction_type = 'view'),
        COUNT(*) FILTER (WHERE interaction_type = 'like'),
        COUNT(*) FILTER (WHERE interaction_type = 'comment'),
        COUNT(*) FILTER (WHERE interaction_type = 'share'),
        COUNT(*) FILTER (WHERE interaction_type = 'save'),
        COUNT(DISTINCT user_id) FILTER (WHERE interaction_type = 'view'),
        NOW()
    FROM post_interactions
    WHERE post_id = NEW.post_id
    ON CONFLICT (post_id) DO UPDATE SET
        view_count = EXCLUDED.view_count,
        like_count = EXCLUDED.like_count,
        comment_count = EXCLUDED.comment_count,
        share_count = EXCLUDED.share_count,
        save_count = EXCLUDED.save_count,
        unique_viewers = EXCLUDED.unique_viewers,
        engagement_rate = CASE
            WHEN EXCLUDED.view_count > 0
            THEN (EXCLUDED.like_count + EXCLUDED.comment_count + EXCLUDED.share_count)::float / EXCLUDED.view_count
            ELSE 0
        END,
        last_updated = NOW();

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update engagement stats on new interactions
CREATE TRIGGER trigger_update_post_engagement_stats
AFTER INSERT ON post_interactions
FOR EACH ROW
EXECUTE FUNCTION update_post_engagement_stats();

-- Function to calculate viral score (can be run periodically)
CREATE OR REPLACE FUNCTION calculate_viral_scores()
RETURNS void AS $$
BEGIN
    UPDATE post_engagement_stats pes
    SET viral_score = (
        SELECT
            (
                like_count * 1.0 +
                comment_count * 3.0 +
                share_count * 5.0
            ) * EXP(-EXTRACT(EPOCH FROM (NOW() - p.created_at)) / 86400.0)  -- Decay over days
        FROM posts p
        WHERE p.id = pes.post_id
    ),
    last_updated = NOW();
END;
$$ LANGUAGE plpgsql;

-- View for trending posts (frequently used query)
CREATE OR REPLACE VIEW trending_posts AS
SELECT
    p.id,
    p.player_id,
    p.content,
    p.media_urls,
    p.hashtags,
    p.created_at,
    CONCAT(pl.first_name, ' ', pl.last_name) as author_name,
    pl.profile_picture as author_avatar,
    pes.like_count,
    pes.comment_count,
    pes.share_count,
    pes.view_count,
    pes.engagement_rate,
    pes.viral_score
FROM posts p
JOIN players pl ON p.player_id = pl.id
LEFT JOIN post_engagement_stats pes ON p.id = pes.post_id
WHERE p.deleted_at IS NULL
    AND p.visibility = 'public'
    AND p.created_at > NOW() - INTERVAL '7 days'
ORDER BY pes.viral_score DESC NULLS LAST;

-- Indexes
CREATE INDEX idx_players_status ON players(status);
CREATE INDEX idx_players_gender ON players(gender);
CREATE INDEX idx_players_birth_date ON players(birth_date);
CREATE INDEX idx_players_height ON players(height);
CREATE INDEX idx_players_weight ON players(weight);
CREATE INDEX idx_players_deleted_at ON players(deleted_at);
CREATE INDEX idx_players_tenant_id ON players(tenant_id);

CREATE INDEX idx_player_skills_player_id ON player_skills(player_id);
CREATE INDEX idx_player_skills_skill ON player_skills(skill);
CREATE INDEX idx_player_skills_level ON player_skills(level);

CREATE INDEX idx_player_teams_player_id ON player_teams(player_id);
CREATE INDEX idx_player_teams_playable ON player_teams(playable_type,playable_id);
CREATE INDEX idx_player_teams_approved ON player_teams(approved_at);
CREATE INDEX idx_player_teams_end_at ON player_teams(end_at);

CREATE INDEX idx_views_viewable ON views(viewable_type,viewable_id);
CREATE INDEX idx_views_viewerable ON views(viewerable_type,viewerable_id);
CREATE INDEX idx_views_created_at ON views(created_at);

CREATE INDEX idx_follows_followerable ON follows(followerable_type,followerable_id);
CREATE INDEX idx_follows_followedable ON follows(followedable_type,followedable_id);

CREATE INDEX idx_chat_messages_room ON chat_messages(chat_room_id);
CREATE INDEX idx_chat_messages_sender ON chat_messages(sender_type,sender_id);
CREATE INDEX idx_chat_messages_created ON chat_messages(created_at);

CREATE INDEX idx_cities_state_id ON cities(state_id);
CREATE INDEX idx_states_country_id ON states(country_id);

CREATE INDEX idx_engagement_events_player_id ON player_engagement_events(player_id);
CREATE INDEX idx_engagement_events_user_id ON player_engagement_events(user_id);
CREATE INDEX idx_engagement_events_created_at ON player_engagement_events(created_at);
CREATE INDEX idx_engagement_events_event_type ON player_engagement_events(event_type);

-- pgvector cosine-distance index (fast ANN search)
-- Using HNSW for better performance than IVFFlat
CREATE INDEX idx_players_embedding_hnsw ON players USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);