theorem RingHom.isIntegralElem_map {x : R} : f.IsIntegralElem (f x) :=
                                /-
                                  R : Type u_1
                                  S : Type u_2
                                  inst✝¹ : CommRing R
                                  inst✝ : Ring S
                                  f : RingHom R S
                                  x : R
                                  ⊢ Eq (Polynomial.eval₂ f (f x) (HSub.hSub Polynomial.X (Polynomial.C x))) 0
                                -/
  ⟨X - C x, monic_X_sub_C _, by simp⟩
                                /-
                                  🎉 no goals
                                -/


theorem isIntegral_algebraMap {x : R} : IsIntegral R (algebraMap R A x) :=
  (algebraMap R A).isIntegralElem_map


theorem IsIntegral.map {B C F : Type*} [Ring B] [Ring C] [Algebra R B] [Algebra A B] [Algebra R C]
    [IsScalarTower R A B] [Algebra A C] [IsScalarTower R A C] {b : B}
    [FunLike F B C] [AlgHomClass F A B C] (f : F)
    (hb : IsIntegral R b) : IsIntegral R (f b) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝¹² : CommRing R
    inst✝¹¹ : CommRing A
    inst✝¹⁰ : Algebra R A
    B : Type u_5
    C : Type u_6
    F : Type u_7
    inst✝⁹ : Ring B
    inst✝⁸ : Ring C
    inst✝⁷ : Algebra R B
    inst✝⁶ : Algebra A B
    inst✝⁵ : Algebra R C
    inst✝⁴ : IsScalarTower R A B
    inst✝³ : Algebra A C
    inst✝² : IsScalarTower R A C
    b : B
    inst✝¹ : FunLike F B C
    inst✝ : AlgHomClass F A B C
    f : F
    hb : IsIntegral R b
    ⊢ IsIntegral R (f b)
  -/
  obtain ⟨P, hP⟩ := hb
  /-
    case intro
    R : Type u_1
    A : Type u_2
    inst✝¹² : CommRing R
    inst✝¹¹ : CommRing A
    inst✝¹⁰ : Algebra R A
    B : Type u_5
    C : Type u_6
    F : Type u_7
    inst✝⁹ : Ring B
    inst✝⁸ : Ring C
    inst✝⁷ : Algebra R B
    inst✝⁶ : Algebra A B
    inst✝⁵ : Algebra R C
    inst✝⁴ : IsScalarTower R A B
    inst✝³ : Algebra A C
    inst✝² : IsScalarTower R A C
    b : B
    inst✝¹ : FunLike F B C
    inst✝ : AlgHomClass F A B C
    f : F
    P : Polynomial R
    hP : And P.Monic (Eq (Polynomial.eval₂ (algebraMap R B) b P) 0)
    ⊢ IsIntegral R (f b)
  -/
  refine ⟨P, hP.1, ?_⟩
  rw [← aeval_def, ← aeval_map_algebraMap A,
    aeval_algHom_apply, aeval_map_algebraMap, aeval_def, hP.2, _root_.map_zero]


theorem isIntegral_algHom_iff (f : A →ₐ[R] B) (hf : Function.Injective f) {x : A} :
    IsIntegral R (f x) ↔ IsIntegral R x := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    A : Type u_5
    B : Type u_6
    inst✝³ : Ring A
    inst✝² : Ring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f : AlgHom R A B
    hf : Function.Injective ⇑f
    x : A
    ⊢ Iff (IsIntegral R (f x)) (IsIntegral R x)
  -/
  refine ⟨fun ⟨p, hp, hx⟩ ↦ ⟨p, hp, ?_⟩, IsIntegral.map f⟩
  rwa [← f.comp_algebraMap, ← AlgHom.coe_toRingHom, ← hom_eval₂, AlgHom.coe_toRingHom,
    map_eq_zero_iff f hf] at hx


open Classical in
theorem Submodule.span_range_natDegree_eq_adjoin {R A} [CommRing R] [Semiring A] [Algebra R A]
    {x : A} {f : R[X]} (hf : f.Monic) (hfx : aeval x f = 0) :
    span R (Finset.image (x ^ ·) (Finset.range (natDegree f))) =
      Subalgebra.toSubmodule (Algebra.adjoin R {x}) := by
  /-
    R : Type u_5
    A : Type u_6
    inst✝² : CommRing R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : A
    f : Polynomial R
    hf : f.Monic
    hfx : Eq ((Polynomial.aeval x) f) 0
    ⊢ Eq (Submodule.span R ↑(Finset.image (fun x_1 => HPow.hPow x x_1) (Finset.ran …
  -/
  nontriviality A
  /-
    R : Type u_5
    A : Type u_6
    inst✝² : CommRing R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : A
    f : Polynomial R
    hf : f.Monic
    hfx : Eq ((Polynomial.aeval x) f) 0
    a✝ : Nontrivial A
    ⊢ Eq (Submodule.span R ↑(Finset.image (fun x_1 => HPow.hPow x x_1) (Finset.ran …
  -/
  have hf1 : f ≠ 1 := by rintro rfl; simp [one_ne_zero' A] at hfx
  /-
    R : Type u_5
    A : Type u_6
    inst✝² : CommRing R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : A
    f : Polynomial R
    hf : f.Monic
    hfx : Eq ((Polynomial.aeval x) f) 0
    a✝ : Nontrivial A
    hf1 : Ne f 1
    ⊢ Eq (Submodule.span R ↑(Finset.image (fun x_1 => HPow.hPow x x_1) (Finset.ran …
  -/
  refine (span_le.mpr fun s hs ↦ ?_).antisymm fun r hr ↦ ?_
    /-
      case refine_1
      R : Type u_5
      A : Type u_6
      inst✝² : CommRing R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      x : A
      f : Polynomial R
      hf : f.Monic
      hfx : Eq ((Polynomial.aeval x) f) 0
      a✝ : Nontrivial A
      hf1 : Ne f 1
      s : A
      hs : Membership.mem (↑(Finset.image (fun x_1 => HPow.hPow x x_1) (Finset.range …
      ⊢ Membership.mem (↑(Subalgebra.toSubmodule (Algebra.adjoin R (Singleton.single …
    -/
  · rcases Finset.mem_image.1 hs with ⟨k, -, rfl⟩
    /-
      case refine_1.intro.intro
      R : Type u_5
      A : Type u_6
      inst✝² : CommRing R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      x : A
      f : Polynomial R
      hf : f.Monic
      hfx : Eq ((Polynomial.aeval x) f) 0
      a✝ : Nontrivial A
      hf1 : Ne f 1
      k : Nat
      hs : Membership.mem (↑(Finset.image (fun x_1 => HPow.hPow x x_1) (Finset.range …
      ⊢ Membership.mem (↑(Subalgebra.toSubmodule (Algebra.adjoin R (Singleton.single …
    -/
    exact (Algebra.adjoin R {x}).pow_mem (Algebra.subset_adjoin rfl) k
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    R : Type u_5
    A : Type u_6
    inst✝² : CommRing R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : A
    f : Polynomial R
    hf : f.Monic
    hfx : Eq ((Polynomial.aeval x) f) 0
    a✝ : Nontrivial A
    hf1 : Ne f 1
    r : A
    hr : Membership.mem (Subalgebra.toSubmodule (Algebra.adjoin R (Singleton.singl …
    ⊢ Membership.mem (Submodule.span R ↑(Finset.image (fun x_1 => HPow.hPow x x_1) …
  -/
  rw [Subalgebra.mem_toSubmodule, Algebra.adjoin_singleton_eq_range_aeval] at hr
  /-
    case refine_2
    R : Type u_5
    A : Type u_6
    inst✝² : CommRing R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : A
    f : Polynomial R
    hf : f.Monic
    hfx : Eq ((Polynomial.aeval x) f) 0
    a✝ : Nontrivial A
    hf1 : Ne f 1
    r : A
    hr : Membership.mem (Polynomial.aeval x).range r
    ⊢ Membership.mem (Submodule.span R ↑(Finset.image (fun x_1 => HPow.hPow x x_1) …
  -/
  rcases (aeval x).mem_range.mp hr with ⟨p, rfl⟩
  rw [← modByMonic_add_div p hf, map_add, map_mul, hfx,
      zero_mul, add_zero, ← sum_C_mul_X_pow_eq (p %ₘ f), aeval_def, eval₂_sum, sum_def]
  /-
    case refine_2.intro
    R : Type u_5
    A : Type u_6
    inst✝² : CommRing R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : A
    f : Polynomial R
    hf : f.Monic
    hfx : Eq ((Polynomial.aeval x) f) 0
    a✝ : Nontrivial A
    hf1 : Ne f 1
    p : Polynomial R
    hr : Membership.mem (Polynomial.aeval x).range ((Polynomial.aeval x) p)
    ⊢ Membership.mem (Submodule.span R ↑(Finset.image (fun x_1 => HPow.hPow x x_1) …
  -/
  refine sum_mem fun k hkq ↦ ?_
  /-
    case refine_2.intro
    R : Type u_5
    A : Type u_6
    inst✝² : CommRing R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : A
    f : Polynomial R
    hf : f.Monic
    hfx : Eq ((Polynomial.aeval x) f) 0
    a✝ : Nontrivial A
    hf1 : Ne f 1
    p : Polynomial R
    hr : Membership.mem (Polynomial.aeval x).range ((Polynomial.aeval x) p)
    k : Nat
    hkq : Membership.mem (p.modByMonic f).support k
    ⊢ Membership.mem (Submodule.span R ↑(Finset.image (fun x_1 => HPow.hPow x x_1) …
  -/
  rw [C_mul_X_pow_eq_monomial, eval₂_monomial, ← Algebra.smul_def]
  exact smul_mem _ _ (subset_span <| Finset.mem_image_of_mem _ <| Finset.mem_range.mpr <|
    (le_natDegree_of_mem_supp _ hkq).trans_lt <| natDegree_modByMonic_lt p hf hf1)


theorem IsIntegral.fg_adjoin_singleton [Algebra R B] {x : B} (hx : IsIntegral R x) :
    (Algebra.adjoin R {x}).toSubmodule.FG := by
  classical
  rcases hx with ⟨f, hfm, hfx⟩
  use (Finset.range <| f.natDegree).image (x ^ ·)
  exact span_range_natDegree_eq_adjoin hfm (by rwa [aeval_def])


theorem RingHom.isIntegralElem_zero : f.IsIntegralElem 0 :=
  f.map_zero ▸ f.isIntegralElem_map


theorem isIntegral_zero [Algebra R B] : IsIntegral R (0 : B) :=
  (algebraMap R B).isIntegralElem_zero


theorem RingHom.isIntegralElem_one : f.IsIntegralElem 1 :=
  f.map_one ▸ f.isIntegralElem_map


theorem isIntegral_one [Algebra R B] : IsIntegral R (1 : B) :=
  (algebraMap R B).isIntegralElem_one


theorem IsIntegral.of_pow [Algebra R B] {x : B} {n : ℕ} (hn : 0 < n) (hx : IsIntegral R <| x ^ n) :
    IsIntegral R x :=
  have ⟨p, hmonic, heval⟩ := hx
                                      /-
                                        R : Type u_1
                                        B : Type u_3
                                        inst✝² : CommRing R
                                        inst✝¹ : Ring B
                                        inst✝ : Algebra R B
                                        x : B
                                        n : Nat
                                        hn : LT.lt 0 n
                                        hx : IsIntegral R (HPow.hPow x n)
                                        p : Polynomial R
                                        hmonic : p.Monic
                                        heval : Eq (Polynomial.eval₂ (algebraMap R B) (HPow.hPow x n) p) 0
                                        ⊢ Eq (Polynomial.eval₂ (algebraMap R B) x ((Polynomial.expand R n) p)) 0
                                      -/
  ⟨expand R n p, hmonic.expand hn, by rwa [← aeval_def, expand_aeval]⟩
                                      /-
                                        🎉 no goals
                                      -/


theorem IsIntegral.of_aeval_monic {x : A} {p : R[X]} (monic : p.Monic)
    (deg : p.natDegree ≠ 0) (hx : IsIntegral R (aeval x p)) : IsIntegral R x :=
  have ⟨p, hmonic, heval⟩ := hx
                                /-
                                  R : Type u_1
                                  A : Type u_2
                                  inst✝² : CommRing R
                                  inst✝¹ : CommRing A
                                  inst✝ : Algebra R A
                                  x : A
                                  p✝ : Polynomial R
                                  monic : p✝.Monic
                                  deg : Ne p✝.natDegree 0
                                  hx : IsIntegral R ((Polynomial.aeval x) p✝)
                                  p : Polynomial R
                                  hmonic : p.Monic
                                  heval : Eq (Polynomial.eval₂ (algebraMap R A) ((Polynomial.aeval x) p✝) p) 0
                                  ⊢ Eq (Polynomial.eval₂ (algebraMap R A) x (p.comp p✝)) 0
                                -/
  ⟨_, hmonic.comp monic deg, by rwa [eval₂_comp, ← aeval_def x]⟩
                                /-
                                  🎉 no goals
                                -/


theorem IsIntegral.map_of_comp_eq {R S T U : Type*} [CommRing R] [Ring S]
    [CommRing T] [Ring U] [Algebra R S] [Algebra T U] (φ : R →+* T) (ψ : S →+* U)
    (h : (algebraMap T U).comp φ = ψ.comp (algebraMap R S)) {a : S} (ha : IsIntegral R a) :
    IsIntegral T (ψ a) :=
  let ⟨p, hp⟩ := ha
  ⟨p.map φ, hp.1.map _, by
    /-
      R : Type u_5
      S : Type u_6
      T : Type u_7
      U : Type u_8
      inst✝⁵ : CommRing R
      inst✝⁴ : Ring S
      inst✝³ : CommRing T
      inst✝² : Ring U
      inst✝¹ : Algebra R S
      inst✝ : Algebra T U
      φ : RingHom R T
      ψ : RingHom S U
      h : Eq ((algebraMap T U).comp φ) (ψ.comp (algebraMap R S))
      a : S
      ha : IsIntegral R a
      p : Polynomial R
      hp : And p.Monic (Eq (Polynomial.eval₂ (algebraMap R S) a p) 0)
      ⊢ Eq (Polynomial.eval₂ (algebraMap T U) (ψ a) (Polynomial.map φ p)) 0
    -/
    rw [← eval_map, map_map, h, ← map_map, eval_map, eval₂_at_apply, eval_map, hp.2, ψ.map_zero]⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem isIntegral_algEquiv {A B : Type*} [Ring A] [Ring B] [Algebra R A] [Algebra R B]
    (f : A ≃ₐ[R] B) {x : A} : IsIntegral R (f x) ↔ IsIntegral R x :=
              /-
                R : Type u_1
                inst✝⁴ : CommRing R
                A : Type u_5
                B : Type u_6
                inst✝³ : Ring A
                inst✝² : Ring B
                inst✝¹ : Algebra R A
                inst✝ : Algebra R B
                f : AlgEquiv R A B
                x : A
                h : IsIntegral R (f x)
                ⊢ IsIntegral R x
              -/
  ⟨fun h ↦ by simpa using h.map f.symm, IsIntegral.map f⟩
              /-
                🎉 no goals
              -/


/-- If `R → A → B` is an algebra tower,
then if the entire tower is an integral extension so is `A → B`. -/
theorem IsIntegral.tower_top [Algebra A B] [IsScalarTower R A B] {x : B}
    (hx : IsIntegral R x) : IsIntegral A x :=
  let ⟨p, hp, hpx⟩ := hx
                                         /-
                                           R : Type u_1
                                           A : Type u_2
                                           B : Type u_3
                                           inst✝⁶ : CommRing R
                                           inst✝⁵ : CommRing A
                                           inst✝⁴ : Ring B
                                           inst✝³ : Algebra R A
                                           inst✝² : Algebra R B
                                           inst✝¹ : Algebra A B
                                           inst✝ : IsScalarTower R A B
                                           x : B
                                           hx : IsIntegral R x
                                           p : Polynomial R
                                           hp : p.Monic
                                           hpx : Eq (Polynomial.eval₂ (algebraMap R B) x p) 0
                                           ⊢ Eq (Polynomial.eval₂ (algebraMap A B) x (Polynomial.map (algebraMap R A) p)) 0
                                         -/
  ⟨p.map <| algebraMap R A, hp.map _, by rw [← aeval_def, aeval_map_algebraMap, aeval_def, hpx]⟩
                                         /-
                                           🎉 no goals
                                         -/

/- If `R` and `T` are isomorphic commutative rings and `S` is an `R`-algebra and a `T`-algebra in
  a compatible way, then an element `a ∈ S` is integral over `R` if and only if it is integral
  over `T`.-/

theorem RingEquiv.isIntegral_iff {R S T : Type*} [CommRing R] [CommRing S] [CommRing T]
    [Algebra R S] [Algebra T S] (φ : R ≃+* T)
    (h : (algebraMap T S).comp φ.toRingHom = algebraMap R S) (a : S) :
    IsIntegral R a ↔ IsIntegral T a := by
  /-
    R : Type u_5
    S : Type u_6
    T : Type u_7
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : CommRing T
    inst✝¹ : Algebra R S
    inst✝ : Algebra T S
    φ : RingEquiv R T
    h : Eq ((algebraMap T S).comp φ.toRingHom) (algebraMap R S)
    a : S
    ⊢ Iff (IsIntegral R a) (IsIntegral T a)
  -/
  constructor <;> intro ha
    /-
      case mp
      R : Type u_5
      S : Type u_6
      T : Type u_7
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra T S
      φ : RingEquiv R T
      h : Eq ((algebraMap T S).comp φ.toRingHom) (algebraMap R S)
      a : S
      ha : IsIntegral R a
      ⊢ IsIntegral T a
    -/
  · letI : Algebra R T := φ.toRingHom.toAlgebra
    letI : IsScalarTower R T S :=
      ⟨fun r t s ↦ by simp only [Algebra.smul_def, map_mul, ← h, mul_assoc]; rfl⟩
    /-
      case mp
      R : Type u_5
      S : Type u_6
      T : Type u_7
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra T S
      φ : RingEquiv R T
      h : Eq ((algebraMap T S).comp φ.toRingHom) (algebraMap R S)
      a : S
      ha : IsIntegral R a
      this✝ : Algebra R T := φ.toRingHom.toAlgebra
      this : IsScalarTower R T S := { smul_assoc := fun r t s => Eq.mpr (id (congr ( …
      ⊢ IsIntegral T a
    -/
    exact IsIntegral.tower_top ha
    /-
      🎉 no goals
    -/
  · have h' : (algebraMap T S) = (algebraMap R S).comp φ.symm.toRingHom := by
      simp only [← h, RingHom.comp_assoc, RingEquiv.toRingHom_eq_coe, RingEquiv.comp_symm,
        RingHomCompTriple.comp_eq]
    /-
      case mpr
      R : Type u_5
      S : Type u_6
      T : Type u_7
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra T S
      φ : RingEquiv R T
      h : Eq ((algebraMap T S).comp φ.toRingHom) (algebraMap R S)
      a : S
      ha : IsIntegral T a
      h' : Eq (algebraMap T S) ((algebraMap R S).comp φ.symm.toRingHom)
      ⊢ IsIntegral R a
    -/
    letI : Algebra T R := φ.symm.toRingHom.toAlgebra
    letI : IsScalarTower T R S :=
      ⟨fun r t s ↦ by simp only [Algebra.smul_def, map_mul, h', mul_assoc]; rfl⟩
    /-
      case mpr
      R : Type u_5
      S : Type u_6
      T : Type u_7
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra T S
      φ : RingEquiv R T
      h : Eq ((algebraMap T S).comp φ.toRingHom) (algebraMap R S)
      a : S
      ha : IsIntegral T a
      h' : Eq (algebraMap T S) ((algebraMap R S).comp φ.symm.toRingHom)
      this✝ : Algebra T R := φ.symm.toRingHom.toAlgebra
      this : IsScalarTower T R S := { smul_assoc := fun r t s => Eq.mpr (id (congr ( …
      ⊢ IsIntegral R a
    -/
    exact IsIntegral.tower_top ha
    /-
      🎉 no goals
    -/


theorem map_isIntegral_int {B C F : Type*} [Ring B] [Ring C] {b : B}
    [FunLike F B C] [RingHomClass F B C] (f : F)
    (hb : IsIntegral ℤ b) : IsIntegral ℤ (f b) :=
  hb.map (f : B →+* C).toIntAlgHom


theorem IsIntegral.of_subring {x : B} (T : Subring R) (hx : IsIntegral T x) : IsIntegral R x :=
  hx.tower_top


protected theorem IsIntegral.algebraMap [Algebra A B] [IsScalarTower R A B] {x : A}
    (h : IsIntegral R x) : IsIntegral R (algebraMap A B x) := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Ring B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    x : A
    h : IsIntegral R x
    ⊢ IsIntegral R ((algebraMap A B) x)
  -/
  rcases h with ⟨f, hf, hx⟩
  /-
    case intro.intro
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Ring B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    x : A
    f : Polynomial R
    hf : f.Monic
    hx : Eq (Polynomial.eval₂ (algebraMap R A) x f) 0
    ⊢ IsIntegral R ((algebraMap A B) x)
  -/
  use f, hf
  /-
    case right
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Ring B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    x : A
    f : Polynomial R
    hf : f.Monic
    hx : Eq (Polynomial.eval₂ (algebraMap R A) x f) 0
    ⊢ Eq (Polynomial.eval₂ (algebraMap R B) ((algebraMap A B) x) f) 0
  -/
  rw [IsScalarTower.algebraMap_eq R A B, ← hom_eval₂, hx, RingHom.map_zero]
  /-
    🎉 no goals
  -/


theorem isIntegral_algebraMap_iff [Algebra A B] [IsScalarTower R A B] {x : A}
    (hAB : Function.Injective (algebraMap A B)) :
    IsIntegral R (algebraMap A B x) ↔ IsIntegral R x :=
  isIntegral_algHom_iff (IsScalarTower.toAlgHom R A B) hAB


theorem isIntegral_iff_isIntegral_closure_finite {r : B} :
    IsIntegral R r ↔ ∃ s : Set R, s.Finite ∧ IsIntegral (Subring.closure s) r := by
  /-
    R : Type u_1
    B : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring B
    inst✝ : Algebra R B
    r : B
    ⊢ Iff (IsIntegral R r) (Exists fun s => And s.Finite (IsIntegral (Subtype fun  …
  -/
  constructor <;> intro hr
    /-
      case mp
      R : Type u_1
      B : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring B
      inst✝ : Algebra R B
      r : B
      hr : IsIntegral R r
      ⊢ Exists fun s => And s.Finite (IsIntegral (Subtype fun x => Membership.mem (S …
    -/
  · rcases hr with ⟨p, hmp, hpr⟩
    /-
      case mp.intro.intro
      R : Type u_1
      B : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring B
      inst✝ : Algebra R B
      r : B
      p : Polynomial R
      hmp : p.Monic
      hpr : Eq (Polynomial.eval₂ (algebraMap R B) r p) 0
      ⊢ Exists fun s => And s.Finite (IsIntegral (Subtype fun x => Membership.mem (S …
    -/
    refine ⟨_, Finset.finite_toSet _, p.restriction, monic_restriction.2 hmp, ?_⟩
    /-
      case mp.intro.intro
      R : Type u_1
      B : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring B
      inst✝ : Algebra R B
      r : B
      p : Polynomial R
      hmp : p.Monic
      hpr : Eq (Polynomial.eval₂ (algebraMap R B) r p) 0
      ⊢ Eq (Polynomial.eval₂ (algebraMap (Subtype fun x => Membership.mem (Subring.c …
    -/
    rw [← aeval_def, ← aeval_map_algebraMap R r p.restriction, map_restriction, aeval_def, hpr]
    /-
      🎉 no goals
    -/
  /-
    case mpr
    R : Type u_1
    B : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring B
    inst✝ : Algebra R B
    r : B
    hr : Exists fun s => And s.Finite (IsIntegral (Subtype fun x => Membership.mem …
    ⊢ IsIntegral R r
  -/
  rcases hr with ⟨s, _, hsr⟩
  /-
    case mpr.intro.intro
    R : Type u_1
    B : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring B
    inst✝ : Algebra R B
    r : B
    s : Set R
    left✝ : s.Finite
    hsr : IsIntegral (Subtype fun x => Membership.mem (Subring.closure s) x) r
    ⊢ IsIntegral R r
  -/
  exact hsr.of_subring _
  /-
    🎉 no goals
  -/


@[stacks 09GH]
theorem fg_adjoin_of_finite {s : Set A} (hfs : s.Finite) (his : ∀ x ∈ s, IsIntegral R x) :
    (Algebra.adjoin R s).toSubmodule.FG :=
  Set.Finite.induction_on hfs
    (fun _ =>
      ⟨{1},
        Submodule.ext fun x => by
          /-
            R : Type u_1
            A : Type u_2
            inst✝² : CommRing R
            inst✝¹ : CommRing A
            inst✝ : Algebra R A
            s : Set A
            hfs : s.Finite
            his : ∀ (x : A), Membership.mem s x → IsIntegral R x
            x✝ : ∀ (x : A), Membership.mem EmptyCollection.emptyCollection x → IsIntegral  …
            x : A
            ⊢ Iff (Membership.mem (Submodule.span R ↑(Singleton.singleton 1)) x) (Membersh …
          -/
          rw [Algebra.adjoin_empty, Finset.coe_singleton, ← one_eq_span, Algebra.toSubmodule_bot]⟩)
          /-
            🎉 no goals
          -/
    (fun {a s} _ _ ih his => by
      /-
        R : Type u_1
        A : Type u_2
        inst✝² : CommRing R
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        s✝ : Set A
        hfs : s✝.Finite
        his✝ : ∀ (x : A), Membership.mem s✝ x → IsIntegral R x
        a : A
        s : Set A
        x✝¹ : Not (Membership.mem s a)
        x✝ : s.Finite
        ih : (∀ (x : A), Membership.mem s x → IsIntegral R x) → (Subalgebra.toSubmodul …
        his : ∀ (x : A), Membership.mem (Insert.insert a s) x → IsIntegral R x
        ⊢ (Subalgebra.toSubmodule (Algebra.adjoin R (Insert.insert a s))).FG
      -/
      rw [← Set.union_singleton, Algebra.adjoin_union_coe_submodule]
      exact
        FG.mul (ih fun i hi => his i <| Set.mem_insert_of_mem a hi)
          (his a <| Set.mem_insert a s).fg_adjoin_singleton)
    his


theorem isNoetherian_adjoin_finset [IsNoetherianRing R] (s : Finset A)
    (hs : ∀ x ∈ s, IsIntegral R x) : IsNoetherian R (Algebra.adjoin R (s : Set A)) :=
  isNoetherian_of_fg_of_noetherian _ (fg_adjoin_of_finite s.finite_toSet hs)


