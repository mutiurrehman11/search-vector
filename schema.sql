CREATE EXTENSION IF NOT EXISTS "uuid-ossp";   -- for UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- for gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "btree_gist"; -- needed for GiST indexes on JSONB
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS cube;
CREATE EXTENSION IF NOT EXISTS earthdistance;

-- Players table – core profile
CREATE TABLE players (
    id          CHAR(26) PRIMARY KEY,
    first_name  VARCHAR NOT NULL,
    last_name   VARCHAR NOT NULL,
    contacts    JSONB,
    description TEXT,
    playing_since DATE,
    height      REAL,
    weight      REAL,
    location    JSONB NOT NULL,
    birth_date  DATE NOT NULL,
    document    VARCHAR,
    profile_picture JSONB,
    pictures    JSONB,
    gender      VARCHAR,
    status      VARCHAR NOT NULL,
    user_id     CHAR(26),
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now(),
    deleted_at  TIMESTAMPTZ
);

-- Player skills
CREATE TABLE player_skills (
    id          BIGSERIAL PRIMARY KEY,
    player_id   CHAR(26) NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    skill       VARCHAR NOT NULL CHECK (skill IN ('dribbling','passing','shooting','speed','stamina','strength')),
    level       INTEGER NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- Player‑team association
CREATE TABLE player_teams (
    id              BIGSERIAL PRIMARY KEY,
    player_id       CHAR(26) NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    playable_type   VARCHAR NOT NULL,
    playable_id     CHAR(26) NOT NULL,
    overview        TEXT,
    start_at        TIMESTAMPTZ NOT NULL,
    end_at          TIMESTAMPTZ,
    position        VARCHAR,
    tshirt          INTEGER,
    approved_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
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
    id              BIGSERIAL PRIMARY KEY,
    ip              VARCHAR NOT NULL,
    viewable_type   VARCHAR NOT NULL,
    viewable_id     CHAR(26) NOT NULL,
    viewerable_type VARCHAR NOT NULL,
    viewerable_id   CHAR(26) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE follows (
    id               BIGSERIAL PRIMARY KEY,
    followerable_type VARCHAR NOT NULL,
    followerable_id   CHAR(26) NOT NULL,
    followedable_type VARCHAR NOT NULL,
    followedable_id   CHAR(26) NOT NULL,
    created_at        TIMESTAMPTZ DEFAULT now(),
    updated_at        TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chat_rooms (
    id          BIGSERIAL PRIMARY KEY,
    name        VARCHAR NOT NULL,
    slug        VARCHAR UNIQUE NOT NULL,
    description TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chat_messages (
    id           BIGSERIAL PRIMARY KEY,
    chat_room_id BIGINT NOT NULL REFERENCES chat_rooms(id) ON DELETE CASCADE,
    sender_type  VARCHAR NOT NULL,
    sender_id    CHAR(26) NOT NULL,
    message      TEXT NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT now(),
    updated_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE playlists (
    id          BIGSERIAL PRIMARY KEY,
    name        VARCHAR NOT NULL,
    description TEXT,
    is_public   BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
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
    name VARCHAR(255) NOT NULL,
    code VARCHAR(10) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);


-- Geographic lookup tables
CREATE TABLE countries (
    id   BIGSERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    code CHAR(3) UNIQUE NOT NULL
);
CREATE TABLE states (
    id         BIGSERIAL PRIMARY KEY,
    name       VARCHAR NOT NULL,
    code       VARCHAR,
    country_id BIGINT NOT NULL REFERENCES countries(id)
);
CREATE TABLE cities (
    id      BIGSERIAL PRIMARY KEY,
    name    VARCHAR NOT NULL,
    state_id BIGINT NOT NULL REFERENCES states(id)
);


-- Store the 128‑dimensional vector as a float array (pgvector extension)
CREATE EXTENSION IF NOT EXISTS vector;   -- `vector` provides the `vector` datatype

CREATE TABLE player_embeddings (
    player_id CHAR(26) PRIMARY KEY REFERENCES players(id) ON DELETE CASCADE,
    embedding vector(128) NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE player_engagement_stats (
    player_id               CHAR(26) PRIMARY KEY REFERENCES players(id) ON DELETE CASCADE,
    ctr                     DOUBLE PRECISION DEFAULT 0,
    recent_activity_score   DOUBLE PRECISION DEFAULT 0,
    follow_rate             DOUBLE PRECISION DEFAULT 0,
    message_rate            DOUBLE PRECISION DEFAULT 0,
    updated_at              TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE player_engagement_events (
    id           BIGSERIAL PRIMARY KEY,
    user_id      CHAR(26) NOT NULL,
    player_id    CHAR(26) NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    event_type   VARCHAR NOT NULL CHECK (event_type IN ('impression','profile_view','follow','message','save_to_playlist')),
    query_context JSONB NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE saved_searches (
    id               BIGSERIAL PRIMARY KEY,
    user_id          CHAR(26) NOT NULL,
    search_name      VARCHAR NOT NULL,
    filters          JSONB NOT NULL,
    query_embedding  vector(128) NOT NULL,
    alert_frequency  VARCHAR NOT NULL DEFAULT 'weekly',
    last_alerted_at  TIMESTAMPTZ,
    created_at       TIMESTAMPTZ DEFAULT now(),
    updated_at       TIMESTAMPTZ DEFAULT now()
);


CREATE INDEX idx_players_status          ON players(status);
CREATE INDEX idx_players_gender          ON players(gender);
CREATE INDEX idx_players_birth_date      ON players(birth_date);
CREATE INDEX idx_players_height          ON players(height);
CREATE INDEX idx_players_weight          ON players(weight);
CREATE INDEX idx_players_deleted_at      ON players(deleted_at);

CREATE INDEX idx_player_skills_player_id ON player_skills(player_id);
CREATE INDEX idx_player_skills_skill     ON player_skills(skill);
CREATE INDEX idx_player_skills_level     ON player_skills(level);

CREATE INDEX idx_player_teams_player_id  ON player_teams(player_id);
CREATE INDEX idx_player_teams_playable   ON player_teams(playable_type,playable_id);
CREATE INDEX idx_player_teams_approved   ON player_teams(approved_at);

CREATE INDEX idx_views_viewable          ON views(viewable_type,viewable_id);
CREATE INDEX idx_views_viewerable        ON views(viewerable_type,viewerable_id);
CREATE INDEX idx_views_created_at        ON views(created_at);

CREATE INDEX idx_follows_followerable    ON follows(followerable_type,followerable_id);
CREATE INDEX idx_follows_followedable    ON follows(followedable_type,followedable_id);

CREATE INDEX idx_chat_messages_room      ON chat_messages(chat_room_id);
CREATE INDEX idx_chat_messages_sender    ON chat_messages(sender_type,sender_id);
CREATE INDEX idx_chat_messages_created   ON chat_messages(created_at);

CREATE INDEX idx_cities_state_id        ON cities(state_id);
CREATE INDEX idx_states_country_id      ON states(country_id);

-- pgvector cosine‑distance index (fast ANN search)
CREATE INDEX idx_player_embeddings_vec ON player_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);