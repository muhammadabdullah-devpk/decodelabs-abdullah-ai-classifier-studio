# Base build stage for compiling native modules
FROM node:20-slim AS builder

WORKDIR /app

# Install compilation tools for native C++ bindings (better-sqlite3)
RUN apt-get update && apt-get install -y \
    python3 \
    make \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy package dependencies and run install
COPY server/package.json ./server/
RUN cd server && npm install --omit=dev

# Production stage
FROM node:20-slim

WORKDIR /app

# Copy compiled node_modules from builder
COPY --from=builder /app/server/node_modules ./server/node_modules

# Copy server and frontend directories
COPY server/ ./server/
COPY public/ ./public/

# Set env and expose ports
ENV PORT=3000
ENV NODE_ENV=production
EXPOSE 3000

WORKDIR /app/server

# Command to launch Express backend and static client
CMD ["node", "index.js"]
