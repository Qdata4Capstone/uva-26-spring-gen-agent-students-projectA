# ── Build stage ──────────────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# ── Production stage ─────────────────────────────────────────
FROM nginx:alpine

ARG BACKEND_HOST=127.0.0.1

COPY --from=builder /app/dist /usr/share/nginx/html

RUN printf '\
worker_processes 1;\n\
pid /tmp/nginx.pid;\n\
error_log /tmp/nginx_error.log warn;\n\
\n\
events {\n\
    worker_connections 1024;\n\
}\n\
\n\
http {\n\
    include       /etc/nginx/mime.types;\n\
    default_type  application/octet-stream;\n\
    access_log    /tmp/nginx_access.log;\n\
\n\
    client_body_temp_path /tmp/nginx_client_body;\n\
    proxy_temp_path       /tmp/nginx_proxy;\n\
    fastcgi_temp_path     /tmp/nginx_fastcgi;\n\
    uwsgi_temp_path       /tmp/nginx_uwsgi;\n\
    scgi_temp_path        /tmp/nginx_scgi;\n\
\n\
    server {\n\
        listen 8080;\n\
\n\
        location / {\n\
            root /usr/share/nginx/html;\n\
            try_files $uri $uri/ /index.html;\n\
        }\n\
\n\
        location /api {\n\
            proxy_pass http://'"${BACKEND_HOST}"':8000;\n\
            proxy_http_version 1.1;\n\
            proxy_set_header Host $host;\n\
            proxy_set_header X-Real-IP $remote_addr;\n\
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n\
            proxy_set_header X-Forwarded-Proto $scheme;\n\
            proxy_buffering off;\n\
            proxy_cache off;\n\
            proxy_read_timeout 300s;\n\
            proxy_send_timeout 300s;\n\
        }\n\
\n\
        location /health {\n\
            proxy_pass http://'"${BACKEND_HOST}"':8000;\n\
            proxy_http_version 1.1;\n\
            proxy_set_header Host $host;\n\
            proxy_buffering off;\n\
            proxy_cache off;\n\
        }\n\
    }\n\
}\n' > /etc/nginx/nginx.conf

EXPOSE 8080

ENTRYPOINT []
CMD ["sh", "-c", "mkdir -p /tmp/nginx_client_body /tmp/nginx_proxy /tmp/nginx_fastcgi /tmp/nginx_uwsgi /tmp/nginx_scgi && exec nginx -g 'daemon off;'"]