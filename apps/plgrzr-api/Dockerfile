# Use Node.js 22 as base image
FROM node:22-slim

# Set working directory
WORKDIR /app

# Install OpenSSL for Prisma
RUN apt-get update -y && \
    apt-get install -y openssl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy package files
COPY package.json pnpm-lock.yaml ./

# Install pnpm
RUN npm install -g pnpm

# Install dependencies
RUN pnpm install

# Copy prisma schema
COPY prisma ./prisma/

# Generate Prisma client
RUN pnpm prisma generate

# Copy the rest of the application
COPY . .

# Expose port 3001
EXPOSE 3001

# Start the application
CMD ["pnpm", "start"]