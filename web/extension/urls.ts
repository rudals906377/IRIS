// 쇼핑몰 이미지 CDN 주소 규칙(서비스 워커와 단위 테스트가 함께 쓴다).

/** 쇼핑몰별로 더 큰 해상도의 이미지 주소로 바꾼다(모르는 사이트는 그대로). */
export function upgradeImageUrl(u: string): string {
  try {
    const url = new URL(u);
    if (url.hostname === 'static.nike.com' && url.pathname.startsWith('/a/images/')) {
      // /a/images/<변환 인자들>/<이미지 id>/<파일명> → 1728px 상세 변환
      // 예: /a/images/t_web_pdp_535_v2/f_auto/abc…/M+NK+TEE.png, /a/images/c_limit,w_592,f_auto/t_product_v1/abc…/x.png
      const seg = url.pathname.split('/').filter(Boolean);
      const idAt = seg.findIndex((s, i) => i >= 2 && /^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$|^[a-z0-9]{20,}$/i.test(s) && i === seg.length - 2);
      if (idAt >= 2) {
        url.pathname = `/a/images/t_PDP_1728_v1/f_auto,q_auto:eco/${seg.slice(idAt).join('/')}`;
        url.search = '';
      }
      return url.href;
    }
    if (url.hostname === 'assets.adidas.com' && url.pathname.startsWith('/images/')) {
      // /images/<변환 인자>/<해시>/<파일명> 또는 /images/<해시>/<파일명> → 가로 1200px
      const seg = url.pathname.split('/').filter(Boolean);
      if (seg.length >= 3) {
        const tail = seg.slice(-2).join('/');
        url.pathname = `/images/w_1200,f_auto,q_auto/${tail}`;
        url.search = '';
      }
      return url.href;
    }
    return u;
  } catch {
    return u;
  }
}
