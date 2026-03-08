#!/usr/bin/env python3
from timelab_core.model_engine import rebuild_target_stats


if __name__ == '__main__':
    doc = rebuild_target_stats()
    print({'rows': len(doc.get('target_stats', []))})