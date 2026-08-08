# Frontend: static files behind nginx. No Python, no build step, no dependencies.
# Build from the repo root: docker build -f docker/frontend.Dockerfile .
FROM nginx:1.27-alpine

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY public/ /usr/share/nginx/html/

EXPOSE 80
