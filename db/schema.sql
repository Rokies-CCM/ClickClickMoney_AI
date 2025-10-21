-- 기존 테이블이 있다면 삭제
DROP TABLE IF EXISTS chat_history;
DROP TABLE IF EXISTS budgets;
DROP TABLE IF EXISTS consumption;
DROP TABLE IF EXISTS categories;
DROP TABLE IF EXISTS users;

-- 사용자 테이블
CREATE TABLE users (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    username VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 카테고리 테이블
CREATE TABLE categories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    category_type ENUM('expense', 'income') NOT NULL -- 'type' -> 'category_type'으로 변경
);

-- 소비 내역 테이블
CREATE TABLE consumption (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    consumption_date DATE NOT NULL, -- 'date' -> 'consumption_date'로 변경
    amount INT NOT NULL,
    details VARCHAR(255),
    category_id INT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

-- 예산 테이블
CREATE TABLE budgets (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    budget_period VARCHAR(7) NOT NULL, -- 'year_month' -> 'budget_period'로 변경
    category_id INT,
    amount INT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

-- AI 챗봇 대화 기록 테이블
CREATE TABLE chat_history (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    session_id VARCHAR(100),
    message JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 기본 데이터 삽입
INSERT INTO users (id, email, password, username) VALUES (1, 'user@example.com', 'hashed_password', '테스트유저');
INSERT INTO categories (id, name, category_type) VALUES
(1, '생활', 'expense'), (2, '식비', 'expense'), (3, '교통', 'expense'),
(4, '주거', 'expense'), (5, '통신', 'expense'), (6, '쇼핑', 'expense'),
(7, '카페/간식', 'expense'), (8, '의료/건강', 'expense'), (9, '문화/여가', 'expense'),
(10, '기타', 'expense');

-- 테스트용 소비 내역 데이터
INSERT INTO consumption (user_id, category_id, consumption_date, amount) VALUES
(1, 2, '2025-10-02', 12500), (1, 7, '2025-10-03', 5500), (1, 3, '2025-10-05', 2400),
(1, 6, '2025-10-07', 78000), (1, 2, '2025-10-09', 21000), (1, 9, '2025-10-11', 45000),
(1, 1, '2025-10-14', 18900), (1, 3, '2025-10-16', 5200), (1, 7, '2025-10-18', 6300),
(1, 8, '2025-10-20', 35000);

-- 인덱스 추가
CREATE INDEX idx_consumption_user_date ON consumption(user_id, consumption_date);
CREATE INDEX idx_budgets_user_month ON budgets(user_id, budget_period);