theorem RingHom.ker_isRadical_iff_reduced_of_surjective {S F} [CommSemiring R] [CommRing S]
    [FunLike F R S] [RingHomClass F R S] {f : F} (hf : Function.Surjective f) :
    (RingHom.ker f).IsRadical ↔ IsReduced S := by
  /-
    R : Type u_1
    S : Type u_3
    F : Type u_4
    inst✝³ : CommSemiring R
    inst✝² : CommRing S
    inst✝¹ : FunLike F R S
    inst✝ : RingHomClass F R S
    f : F
    hf : Function.Surjective ⇑f
    ⊢ Iff (RingHom.ker f).IsRadical (IsReduced S)
  -/
  simp_rw [isReduced_iff, hf.forall, IsNilpotent, ← map_pow, ← RingHom.mem_ker]
  /-
    R : Type u_1
    S : Type u_3
    F : Type u_4
    inst✝³ : CommSemiring R
    inst✝² : CommRing S
    inst✝¹ : FunLike F R S
    inst✝ : RingHomClass F R S
    f : F
    hf : Function.Surjective ⇑f
    ⊢ Iff (RingHom.ker f).IsRadical (∀ (x : R), (Exists fun n => Membership.mem (R …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem isRadical_iff_span_singleton [CommSemiring R] :
    IsRadical y ↔ (Ideal.span ({y} : Set R)).IsRadical := by
  /-
    R : Type u_1
    y : R
    inst✝ : CommSemiring R
    ⊢ Iff (IsRadical y) (Ideal.span (Singleton.singleton y)).IsRadical
  -/
  simp_rw [IsRadical, ← Ideal.mem_span_singleton]
  /-
    R : Type u_1
    y : R
    inst✝ : CommSemiring R
    ⊢ Iff (∀ (n : Nat) (x : R), Membership.mem (Ideal.span (Singleton.singleton y) …
  -/
  exact forall_swap.trans (forall_congr' fun r => exists_imp.symm)
  /-
    🎉 no goals
  -/


/-- The nilradical of a commutative semiring is the ideal of nilpotent elements. -/
def nilradical (R : Type*) [CommSemiring R] : Ideal R :=
  (0 : Ideal R).radical


theorem mem_nilradical : x ∈ nilradical R ↔ IsNilpotent x :=
  Iff.rfl


theorem nilradical_eq_sInf (R : Type*) [CommSemiring R] :
    nilradical R = sInf { J : Ideal R | J.IsPrime } :=
                                        /-
                                          R : Type u_3
                                          inst✝ : CommSemiring R
                                          ⊢ Eq (InfSet.sInf (setOf fun J => And (LE.le Bot.bot J) J.IsPrime)) (InfSet.sI …
                                        -/
  (Ideal.radical_eq_sInf ⊥).trans <| by simp_rw [and_iff_right bot_le]
                                        /-
                                          🎉 no goals
                                        -/


theorem nilpotent_iff_mem_prime : IsNilpotent x ↔ ∀ J : Ideal R, J.IsPrime → x ∈ J := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    x : R
    ⊢ Iff (IsNilpotent x) (∀ (J : Ideal R), J.IsPrime → Membership.mem J x)
  -/
  rw [← mem_nilradical, nilradical_eq_sInf, Submodule.mem_sInf]
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    x : R
    ⊢ Iff (∀ (p : Submodule R R), Membership.mem (setOf fun J => J.IsPrime) p → Me …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem nilradical_le_prime (J : Ideal R) [H : J.IsPrime] : nilradical R ≤ J :=
  (nilradical_eq_sInf R).symm ▸ sInf_le H


@[simp]
theorem nilradical_eq_zero (R : Type*) [CommSemiring R] [IsReduced R] : nilradical R = 0 :=
  Ideal.ext fun _ => isNilpotent_iff_eq_zero


@[simp]
theorem isNilpotent_mulLeft_iff (a : A) : IsNilpotent (mulLeft R a) ↔ IsNilpotent a := by
  /-
    R : Type u_1
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    a : A
    ⊢ Iff (IsNilpotent (LinearMap.mulLeft R a)) (IsNilpotent a)
  -/
  constructor <;> rintro ⟨n, hn⟩ <;> use n <;>
      /-
        case h
        R : Type u_1
        A : Type v
        inst✝² : CommSemiring R
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        a : A
        n : Nat
        hn : Eq (HPow.hPow (LinearMap.mulLeft R a) n) 0
        ⊢ Eq (HPow.hPow a n) 0
      -/
      simp only [mulLeft_eq_zero_iff, pow_mulLeft] at hn ⊢ <;>
    /-
      case h
      R : Type u_1
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      a : A
      n : Nat
      hn : Eq (HPow.hPow a n) 0
      ⊢ Eq (HPow.hPow a n) 0
    -/
    /-
      🎉 no goals
    -/
    exact hn
    /-
      🎉 no goals
    -/


@[simp]
theorem isNilpotent_mulRight_iff (a : A) : IsNilpotent (mulRight R a) ↔ IsNilpotent a := by
  /-
    R : Type u_1
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    a : A
    ⊢ Iff (IsNilpotent (LinearMap.mulRight R a)) (IsNilpotent a)
  -/
  constructor <;> rintro ⟨n, hn⟩ <;> use n <;>
      /-
        case h
        R : Type u_1
        A : Type v
        inst✝² : CommSemiring R
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        a : A
        n : Nat
        hn : Eq (HPow.hPow (LinearMap.mulRight R a) n) 0
        ⊢ Eq (HPow.hPow a n) 0
      -/
      simp only [mulRight_eq_zero_iff, pow_mulRight] at hn ⊢ <;>
    /-
      case h
      R : Type u_1
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      a : A
      n : Nat
      hn : Eq (HPow.hPow a n) 0
      ⊢ Eq (HPow.hPow a n) 0
    -/
    /-
      🎉 no goals
    -/
    exact hn
    /-
      🎉 no goals
    -/


@[simp]
lemma isNilpotent_toMatrix_iff (b : Basis ι R M) (f : M →ₗ[R] M) :
    IsNilpotent (toMatrix b b f) ↔ IsNilpotent f := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    ι : Type u_3
    M : Type u_4
    inst✝³ : Fintype ι
    inst✝² : DecidableEq ι
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    f : LinearMap (RingHom.id R) M M
    ⊢ Iff (IsNilpotent ((LinearMap.toMatrix b b) f)) (IsNilpotent f)
  -/
  refine exists_congr fun k ↦ ?_
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    ι : Type u_3
    M : Type u_4
    inst✝³ : Fintype ι
    inst✝² : DecidableEq ι
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    f : LinearMap (RingHom.id R) M M
    k : Nat
    ⊢ Iff (Eq (HPow.hPow ((LinearMap.toMatrix b b) f) k) 0) (Eq (HPow.hPow f k) 0)
  -/
  rw [toMatrix_pow]
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    ι : Type u_3
    M : Type u_4
    inst✝³ : Fintype ι
    inst✝² : DecidableEq ι
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    f : LinearMap (RingHom.id R) M M
    k : Nat
    ⊢ Iff (Eq ((LinearMap.toMatrix b b) (HPow.hPow f k)) 0) (Eq (HPow.hPow f k) 0)
  -/
  exact (toMatrix b b).map_eq_zero_iff
  /-
    🎉 no goals
  -/


lemma isNilpotent_restrict_of_le {f : End R M} {p q : Submodule R M}
    {hp : MapsTo f p p} {hq : MapsTo f q q} (h : p ≤ q) (hf : IsNilpotent (f.restrict hq)) :
    IsNilpotent (f.restrict hp) := by
  /-
    R : Type u_1
    M : Type u_3
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p q : Submodule R M
    hp : Set.MapsTo ⇑f ↑p ↑p
    hq : Set.MapsTo ⇑f ↑q ↑q
    h : LE.le p q
    hf : IsNilpotent (LinearMap.restrict f hq)
    ⊢ IsNilpotent (LinearMap.restrict f hp)
  -/
  obtain ⟨n, hn⟩ := hf
  /-
    case intro
    R : Type u_1
    M : Type u_3
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p q : Submodule R M
    hp : Set.MapsTo ⇑f ↑p ↑p
    hq : Set.MapsTo ⇑f ↑q ↑q
    h : LE.le p q
    n : Nat
    hn : Eq (HPow.hPow (LinearMap.restrict f hq) n) 0
    ⊢ IsNilpotent (LinearMap.restrict f hp)
  -/
  use n
  /-
    case h
    R : Type u_1
    M : Type u_3
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p q : Submodule R M
    hp : Set.MapsTo ⇑f ↑p ↑p
    hq : Set.MapsTo ⇑f ↑q ↑q
    h : LE.le p q
    n : Nat
    hn : Eq (HPow.hPow (LinearMap.restrict f hq) n) 0
    ⊢ Eq (HPow.hPow (LinearMap.restrict f hp) n) 0
  -/
  ext ⟨x, hx⟩
  /-
    case h.h.mk.a
    R : Type u_1
    M : Type u_3
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p q : Submodule R M
    hp : Set.MapsTo ⇑f ↑p ↑p
    hq : Set.MapsTo ⇑f ↑q ↑q
    h : LE.le p q
    n : Nat
    hn : Eq (HPow.hPow (LinearMap.restrict f hq) n) 0
    x : M
    hx : Membership.mem p x
    ⊢ Eq ↑((HPow.hPow (LinearMap.restrict f hp) n) ⟨x, hx⟩) ↑(0 ⟨x, hx⟩)
  -/
  replace hn := DFunLike.congr_fun hn ⟨x, h hx⟩
  /-
    case h.h.mk.a
    R : Type u_1
    M : Type u_3
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p q : Submodule R M
    hp : Set.MapsTo ⇑f ↑p ↑p
    hq : Set.MapsTo ⇑f ↑q ↑q
    h : LE.le p q
    n : Nat
    x : M
    hx : Membership.mem p x
    hn : Eq ((HPow.hPow (LinearMap.restrict f hq) n) ⟨x, ⋯⟩) (0 ⟨x, ⋯⟩)
    ⊢ Eq ↑((HPow.hPow (LinearMap.restrict f hp) n) ⟨x, hx⟩) ↑(0 ⟨x, hx⟩)
  -/
  simp_rw [LinearMap.zero_apply, ZeroMemClass.coe_zero, ZeroMemClass.coe_eq_zero] at hn ⊢
  /-
    case h.h.mk.a
    R : Type u_1
    M : Type u_3
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p q : Submodule R M
    hp : Set.MapsTo ⇑f ↑p ↑p
    hq : Set.MapsTo ⇑f ↑q ↑q
    h : LE.le p q
    n : Nat
    x : M
    hx : Membership.mem p x
    hn : Eq ((HPow.hPow (LinearMap.restrict f hq) n) ⟨x, ⋯⟩) 0
    ⊢ Eq ((HPow.hPow (LinearMap.restrict f hp) n) ⟨x, hx⟩) 0
  -/
  rw [LinearMap.pow_restrict, LinearMap.restrict_apply] at hn ⊢
  /-
    case h.h.mk.a
    R : Type u_1
    M : Type u_3
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p q : Submodule R M
    hp : Set.MapsTo ⇑f ↑p ↑p
    hq : Set.MapsTo ⇑f ↑q ↑q
    h : LE.le p q
    n : Nat
    x : M
    hx : Membership.mem p x
    hn : Eq ⟨(HPow.hPow f n) ↑⟨x, ⋯⟩, ⋯⟩ 0
    ⊢ Eq ⟨(HPow.hPow f n) ↑⟨x, hx⟩, ⋯⟩ 0
  -/
  ext
  /-
    case h.h.mk.a.a
    R : Type u_1
    M : Type u_3
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p q : Submodule R M
    hp : Set.MapsTo ⇑f ↑p ↑p
    hq : Set.MapsTo ⇑f ↑q ↑q
    h : LE.le p q
    n : Nat
    x : M
    hx : Membership.mem p x
    hn : Eq ⟨(HPow.hPow f n) ↑⟨x, ⋯⟩, ⋯⟩ 0
    ⊢ Eq ↑⟨(HPow.hPow f n) ↑⟨x, hx⟩, ⋯⟩ ↑0
  -/
  exact (congr_arg Subtype.val hn : _)
  /-
    🎉 no goals
  -/


lemma isNilpotent.restrict
    {f : M →ₗ[R] M} {p : Submodule R M} (hf : MapsTo f p p) (hnil : IsNilpotent f) :
    IsNilpotent (f.restrict hf) := by
  /-
    R : Type u_1
    M : Type u_3
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p : Submodule R M
    hf : Set.MapsTo ⇑f ↑p ↑p
    hnil : IsNilpotent f
    ⊢ IsNilpotent (f.restrict hf)
  -/
  obtain ⟨n, hn⟩ := hnil
  exact ⟨n, LinearMap.ext fun m ↦ by simp only [LinearMap.pow_restrict n, hn,
    LinearMap.restrict_apply, LinearMap.zero_apply]; rfl⟩


theorem IsNilpotent.mapQ (hnp : IsNilpotent f) : IsNilpotent (p.mapQ p f hp) := by
  /-
    R : Type u_1
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : LE.le p (Submodule.comap f p)
    hnp : IsNilpotent f
    ⊢ IsNilpotent (p.mapQ p f hp)
  -/
  obtain ⟨k, hk⟩ := hnp
  /-
    case intro
    R : Type u_1
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : LE.le p (Submodule.comap f p)
    k : Nat
    hk : Eq (HPow.hPow f k) 0
    ⊢ IsNilpotent (p.mapQ p f hp)
  -/
  use k
  /-
    case h
    R : Type u_1
    M : Type v
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : LE.le p (Submodule.comap f p)
    k : Nat
    hk : Eq (HPow.hPow f k) 0
    ⊢ Eq (HPow.hPow (p.mapQ p f hp) k) 0
  -/
  simp [← p.mapQ_pow, hk]
  /-
    🎉 no goals
  -/


