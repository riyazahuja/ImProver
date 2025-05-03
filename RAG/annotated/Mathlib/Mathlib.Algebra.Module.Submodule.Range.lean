/-- The range of a linear map `f : M → M₂` is a submodule of `M₂`.
See Note [range copy pattern]. -/
def range [RingHomSurjective τ₁₂] (f : F) : Submodule R₂ M₂ :=
  (map f ⊤).copy (Set.range f) Set.image_univ.symm


theorem range_coe [RingHomSurjective τ₁₂] (f : F) : (range f : Set M₂) = Set.range f :=
  rfl


theorem range_toAddSubmonoid [RingHomSurjective τ₁₂] (f : M →ₛₗ[τ₁₂] M₂) :
    f.range.toAddSubmonoid = AddMonoidHom.mrange f :=
  rfl


@[simp]
theorem mem_range [RingHomSurjective τ₁₂] {f : F} {x} : x ∈ range f ↔ ∃ y, f y = x :=
  Iff.rfl


theorem range_eq_map [RingHomSurjective τ₁₂] (f : F) : range f = map f ⊤ := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_6
    inst✝⁸ : Semiring R
    inst✝⁷ : Semiring R₂
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M
    inst✝³ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    F : Type u_10
    inst✝² : FunLike F M M₂
    inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
    inst✝ : RingHomSurjective τ₁₂
    f : F
    ⊢ Eq (LinearMap.range f) (Submodule.map f Top.top)
  -/
  ext
  /-
    case h
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_6
    inst✝⁸ : Semiring R
    inst✝⁷ : Semiring R₂
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M
    inst✝³ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    F : Type u_10
    inst✝² : FunLike F M M₂
    inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
    inst✝ : RingHomSurjective τ₁₂
    f : F
    x✝ : M₂
    ⊢ Iff (Membership.mem (LinearMap.range f) x✝) (Membership.mem (Submodule.map f …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem mem_range_self [RingHomSurjective τ₁₂] (f : F) (x : M) : f x ∈ range f :=
  ⟨x, rfl⟩


@[simp]
theorem range_id : range (LinearMap.id : M →ₗ[R] M) = ⊤ :=
  SetLike.coe_injective Set.range_id


theorem range_comp [RingHomSurjective τ₁₂] [RingHomSurjective τ₂₃] [RingHomSurjective τ₁₃]
    (f : M →ₛₗ[τ₁₂] M₂) (g : M₂ →ₛₗ[τ₂₃] M₃) : range (g.comp f : M →ₛₗ[τ₁₃] M₃) = map g (range f) :=
  SetLike.coe_injective (Set.range_comp g f)


theorem range_comp_le_range [RingHomSurjective τ₂₃] [RingHomSurjective τ₁₃] (f : M →ₛₗ[τ₁₂] M₂)
    (g : M₂ →ₛₗ[τ₂₃] M₃) : range (g.comp f : M →ₛₗ[τ₁₃] M₃) ≤ range g :=
  SetLike.coe_mono (Set.range_comp_subset_range f g)


theorem range_eq_top [RingHomSurjective τ₁₂] {f : F} :
    range f = ⊤ ↔ Surjective f := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_6
    inst✝⁸ : Semiring R
    inst✝⁷ : Semiring R₂
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M
    inst✝³ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    F : Type u_10
    inst✝² : FunLike F M M₂
    inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
    inst✝ : RingHomSurjective τ₁₂
    f : F
    ⊢ Iff (Eq (LinearMap.range f) Top.top) (Function.Surjective ⇑f)
  -/
  rw [SetLike.ext'_iff, range_coe, top_coe, Set.range_eq_univ]
  /-
    🎉 no goals
  -/


theorem range_eq_top_of_surjective [RingHomSurjective τ₁₂] (f : F) (hf : Surjective f) :
    range f = ⊤ := range_eq_top.2 hf


theorem range_le_iff_comap [RingHomSurjective τ₁₂] {f : F} {p : Submodule R₂ M₂} :
                                      /-
                                        R : Type u_1
                                        R₂ : Type u_2
                                        M : Type u_5
                                        M₂ : Type u_6
                                        inst✝⁸ : Semiring R
                                        inst✝⁷ : Semiring R₂
                                        inst✝⁶ : AddCommMonoid M
                                        inst✝⁵ : AddCommMonoid M₂
                                        inst✝⁴ : Module R M
                                        inst✝³ : Module R₂ M₂
                                        τ₁₂ : RingHom R R₂
                                        F : Type u_10
                                        inst✝² : FunLike F M M₂
                                        inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
                                        inst✝ : RingHomSurjective τ₁₂
                                        f : F
                                        p : Submodule R₂ M₂
                                        ⊢ Iff (LE.le (LinearMap.range f) p) (Eq (Submodule.comap f p) Top.top)
                                      -/
    range f ≤ p ↔ comap f p = ⊤ := by rw [range_eq_map, map_le_iff_le_comap, eq_top_iff]
                                      /-
                                        🎉 no goals
                                      -/


theorem map_le_range [RingHomSurjective τ₁₂] {f : F} {p : Submodule R M} : map f p ≤ range f :=
  SetLike.coe_mono (Set.image_subset_range f p)


@[simp]
theorem range_neg {R : Type*} {R₂ : Type*} {M : Type*} {M₂ : Type*} [Semiring R] [Ring R₂]
    [AddCommMonoid M] [AddCommGroup M₂] [Module R M] [Module R₂ M₂] {τ₁₂ : R →+* R₂}
    [RingHomSurjective τ₁₂] (f : M →ₛₗ[τ₁₂] M₂) : LinearMap.range (-f) = LinearMap.range f := by
  /-
    R : Type u_11
    R₂ : Type u_12
    M : Type u_13
    M₂ : Type u_14
    inst✝⁶ : Semiring R
    inst✝⁵ : Ring R₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    ⊢ Eq (LinearMap.range (Neg.neg f)) (LinearMap.range f)
  -/
  change range ((-LinearMap.id : M₂ →ₗ[R₂] M₂).comp f) = _
  /-
    R : Type u_11
    R₂ : Type u_12
    M : Type u_13
    M₂ : Type u_14
    inst✝⁶ : Semiring R
    inst✝⁵ : Ring R₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    ⊢ Eq (LinearMap.range ((Neg.neg LinearMap.id).comp f)) (LinearMap.range f)
  -/
  rw [range_comp, Submodule.map_neg, Submodule.map_id]
  /-
    🎉 no goals
  -/


@[simp] lemma range_domRestrict [Module R M₂] (K : Submodule R M) (f : M →ₗ[R] M₂) :
                                            /-
                                              R : Type u_1
                                              M : Type u_5
                                              M₂ : Type u_6
                                              inst✝⁴ : Semiring R
                                              inst✝³ : AddCommMonoid M
                                              inst✝² : AddCommMonoid M₂
                                              inst✝¹ : Module R M
                                              inst✝ : Module R M₂
                                              K : Submodule R M
                                              f : LinearMap (RingHom.id R) M M₂
                                              ⊢ Eq (LinearMap.range (f.domRestrict K)) (Submodule.map f K)
                                            -/
    range (domRestrict f K) = K.map f := by ext; simp
                                                 /-
                                                   🎉 no goals
                                                 -/


lemma range_domRestrict_le_range [RingHomSurjective τ₁₂] (f : M →ₛₗ[τ₁₂] M₂) (S : Submodule R M) :
    LinearMap.range (f.domRestrict S) ≤ LinearMap.range f := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_6
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring R₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    S : Submodule R M
    ⊢ LE.le (LinearMap.range (f.domRestrict S)) (LinearMap.range f)
  -/
  rintro x ⟨⟨y, hy⟩, rfl⟩
  /-
    case intro.mk
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_6
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring R₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    S : Submodule R M
    y : M
    hy : Membership.mem S y
    ⊢ Membership.mem (LinearMap.range f) ((f.domRestrict S) ⟨y, hy⟩)
  -/
  exact LinearMap.mem_range_self f y
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.AddMonoidHom.coe_toIntLinearMap_range {M M₂ : Type*} [AddCommGroup M]
    [AddCommGroup M₂] (f : M →+ M₂) :
    LinearMap.range f.toIntLinearMap = AddSubgroup.toIntSubmodule f.range := rfl


lemma _root_.Submodule.map_comap_eq_of_le [RingHomSurjective τ₁₂] {f : F} {p : Submodule R₂ M₂}
    (h : p ≤ LinearMap.range f) : (p.comap f).map f = p :=
  SetLike.coe_injective <| Set.image_preimage_eq_of_subset h


/-- The decreasing sequence of submodules consisting of the ranges of the iterates of a linear map.
-/
@[simps]
def iterateRange (f : M →ₗ[R] M) : ℕ →o (Submodule R M)ᵒᵈ where
  toFun n := LinearMap.range (f ^ n)
  monotone' n m w x h := by
    /-
      R : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      K : Type u_4
      M : Type u_5
      M₂ : Type u_6
      M₃ : Type u_7
      V : Type u_8
      V₂ : Type u_9
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring R₂
      inst✝⁷ : Semiring R₃
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : AddCommMonoid M₃
      inst✝³ : Module R M
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      τ₁₂ : RingHom R R₂
      τ₂₃ : RingHom R₂ R₃
      τ₁₃ : RingHom R R₃
      inst✝ : RingHomCompTriple τ₁₂ τ₂₃ τ₁₃
      f : LinearMap (RingHom.id R) M M
      n m : Nat
      w : LE.le n m
      x : M
      h : Membership.mem ((fun n => LinearMap.range (HPow.hPow f n)) m) x
      ⊢ Membership.mem ((fun n => LinearMap.range (HPow.hPow f n)) n) x
    -/
    obtain ⟨c, rfl⟩ := Nat.exists_eq_add_of_le  w
    /-
      case intro
      R : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      K : Type u_4
      M : Type u_5
      M₂ : Type u_6
      M₃ : Type u_7
      V : Type u_8
      V₂ : Type u_9
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring R₂
      inst✝⁷ : Semiring R₃
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : AddCommMonoid M₃
      inst✝³ : Module R M
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      τ₁₂ : RingHom R R₂
      τ₂₃ : RingHom R₂ R₃
      τ₁₃ : RingHom R R₃
      inst✝ : RingHomCompTriple τ₁₂ τ₂₃ τ₁₃
      f : LinearMap (RingHom.id R) M M
      n : Nat
      x : M
      c : Nat
      w : LE.le n (HAdd.hAdd n c)
      h : Membership.mem ((fun n => LinearMap.range (HPow.hPow f n)) (HAdd.hAdd n c) …
      ⊢ Membership.mem ((fun n => LinearMap.range (HPow.hPow f n)) n) x
    -/
    rw [LinearMap.mem_range] at h
    /-
      case intro
      R : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      K : Type u_4
      M : Type u_5
      M₂ : Type u_6
      M₃ : Type u_7
      V : Type u_8
      V₂ : Type u_9
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring R₂
      inst✝⁷ : Semiring R₃
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : AddCommMonoid M₃
      inst✝³ : Module R M
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      τ₁₂ : RingHom R R₂
      τ₂₃ : RingHom R₂ R₃
      τ₁₃ : RingHom R R₃
      inst✝ : RingHomCompTriple τ₁₂ τ₂₃ τ₁₃
      f : LinearMap (RingHom.id R) M M
      n : Nat
      x : M
      c : Nat
      w : LE.le n (HAdd.hAdd n c)
      h : Exists fun y => Eq ((HPow.hPow f (HAdd.hAdd n c)) y) x
      ⊢ Membership.mem ((fun n => LinearMap.range (HPow.hPow f n)) n) x
    -/
    obtain ⟨m, rfl⟩ := h
    /-
      case intro.intro
      R : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      K : Type u_4
      M : Type u_5
      M₂ : Type u_6
      M₃ : Type u_7
      V : Type u_8
      V₂ : Type u_9
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring R₂
      inst✝⁷ : Semiring R₃
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : AddCommMonoid M₃
      inst✝³ : Module R M
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      τ₁₂ : RingHom R R₂
      τ₂₃ : RingHom R₂ R₃
      τ₁₃ : RingHom R R₃
      inst✝ : RingHomCompTriple τ₁₂ τ₂₃ τ₁₃
      f : LinearMap (RingHom.id R) M M
      n c : Nat
      w : LE.le n (HAdd.hAdd n c)
      m : M
      ⊢ Membership.mem ((fun n => LinearMap.range (HPow.hPow f n)) n) ((HPow.hPow f  …
    -/
    rw [LinearMap.mem_range]
    /-
      case intro.intro
      R : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      K : Type u_4
      M : Type u_5
      M₂ : Type u_6
      M₃ : Type u_7
      V : Type u_8
      V₂ : Type u_9
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring R₂
      inst✝⁷ : Semiring R₃
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : AddCommMonoid M₃
      inst✝³ : Module R M
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      τ₁₂ : RingHom R R₂
      τ₂₃ : RingHom R₂ R₃
      τ₁₃ : RingHom R R₃
      inst✝ : RingHomCompTriple τ₁₂ τ₂₃ τ₁₃
      f : LinearMap (RingHom.id R) M M
      n c : Nat
      w : LE.le n (HAdd.hAdd n c)
      m : M
      ⊢ Exists fun y => Eq ((HPow.hPow f n) y) ((HPow.hPow f (HAdd.hAdd n c)) m)
    -/
    use (f ^ c) m
    /-
      case h
      R : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      K : Type u_4
      M : Type u_5
      M₂ : Type u_6
      M₃ : Type u_7
      V : Type u_8
      V₂ : Type u_9
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring R₂
      inst✝⁷ : Semiring R₃
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : AddCommMonoid M₃
      inst✝³ : Module R M
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      τ₁₂ : RingHom R R₂
      τ₂₃ : RingHom R₂ R₃
      τ₁₃ : RingHom R R₃
      inst✝ : RingHomCompTriple τ₁₂ τ₂₃ τ₁₃
      f : LinearMap (RingHom.id R) M M
      n c : Nat
      w : LE.le n (HAdd.hAdd n c)
      m : M
      ⊢ Eq ((HPow.hPow f n) ((HPow.hPow f c) m)) ((HPow.hPow f (HAdd.hAdd n c)) m)
    -/
    rw [pow_add, LinearMap.mul_apply]
    /-
      🎉 no goals
    -/


/-- Restrict the codomain of a linear map `f` to `f.range`.

This is the bundled version of `Set.rangeFactorization`. -/
abbrev rangeRestrict [RingHomSurjective τ₁₂] (f : M →ₛₗ[τ₁₂] M₂) : M →ₛₗ[τ₁₂] LinearMap.range f :=
  f.codRestrict (LinearMap.range f) (LinearMap.mem_range_self f)


/-- The range of a linear map is finite if the domain is finite.
Note: this instance can form a diamond with `Subtype.fintype` in the
  presence of `Fintype M₂`. -/
instance fintypeRange [Fintype M] [DecidableEq M₂] [RingHomSurjective τ₁₂] (f : M →ₛₗ[τ₁₂] M₂) :
    Fintype (range f) :=
  Set.fintypeRange f


theorem range_codRestrict {τ₂₁ : R₂ →+* R} [RingHomSurjective τ₂₁] (p : Submodule R M)
    (f : M₂ →ₛₗ[τ₂₁] M) (hf) :
    range (codRestrict p f hf) = comap p.subtype (LinearMap.range f) := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_6
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring R₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M
    inst✝¹ : Module R₂ M₂
    τ₂₁ : RingHom R₂ R
    inst✝ : RingHomSurjective τ₂₁
    p : Submodule R M
    f : LinearMap τ₂₁ M₂ M
    hf : ∀ (c : M₂), Membership.mem p (f c)
    ⊢ Eq (LinearMap.range (LinearMap.codRestrict p f hf)) (Submodule.comap p.subty …
  -/
  simpa only [range_eq_map] using map_codRestrict _ _ _ _
  /-
    🎉 no goals
  -/


theorem _root_.Submodule.map_comap_eq [RingHomSurjective τ₁₂] (f : F) (q : Submodule R₂ M₂) :
    map f (comap f q) = range f ⊓ q :=
  le_antisymm (le_inf map_le_range (map_comap_le _ _)) <| by
    /-
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_6
      inst✝⁸ : Semiring R
      inst✝⁷ : Semiring R₂
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : Module R M
      inst✝³ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      F : Type u_10
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
      inst✝ : RingHomSurjective τ₁₂
      f : F
      q : Submodule R₂ M₂
      ⊢ LE.le (Min.min (LinearMap.range f) q) (Submodule.map f (Submodule.comap f q))
    -/
    rintro _ ⟨⟨x, _, rfl⟩, hx⟩; exact ⟨x, hx, rfl⟩
                                /-
                                  🎉 no goals
                                -/


theorem _root_.Submodule.map_comap_eq_self [RingHomSurjective τ₁₂] {f : F} {q : Submodule R₂ M₂}
                                                    /-
                                                      R : Type u_1
                                                      R₂ : Type u_2
                                                      M : Type u_5
                                                      M₂ : Type u_6
                                                      inst✝⁸ : Semiring R
                                                      inst✝⁷ : Semiring R₂
                                                      inst✝⁶ : AddCommMonoid M
                                                      inst✝⁵ : AddCommMonoid M₂
                                                      inst✝⁴ : Module R M
                                                      inst✝³ : Module R₂ M₂
                                                      τ₁₂ : RingHom R R₂
                                                      F : Type u_10
                                                      inst✝² : FunLike F M M₂
                                                      inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
                                                      inst✝ : RingHomSurjective τ₁₂
                                                      f : F
                                                      q : Submodule R₂ M₂
                                                      h : LE.le q (LinearMap.range f)
                                                      ⊢ Eq (Submodule.map f (Submodule.comap f q)) q
                                                    -/
    (h : q ≤ range f) : map f (comap f q) = q := by rwa [Submodule.map_comap_eq, inf_eq_right]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem range_zero [RingHomSurjective τ₁₂] : range (0 : M →ₛₗ[τ₁₂] M₂) = ⊥ := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_6
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring R₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    ⊢ Eq (LinearMap.range 0) Bot.bot
  -/
  simpa only [range_eq_map] using Submodule.map_zero _
  /-
    🎉 no goals
  -/


theorem range_le_bot_iff (f : M →ₛₗ[τ₁₂] M₂) : range f ≤ ⊥ ↔ f = 0 := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_6
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring R₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    ⊢ Iff (LE.le (LinearMap.range f) Bot.bot) (Eq f 0)
  -/
  rw [range_le_iff_comap]; exact ker_eq_top
                           /-
                             🎉 no goals
                           -/


theorem range_eq_bot {f : M →ₛₗ[τ₁₂] M₂} : range f = ⊥ ↔ f = 0 := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_6
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring R₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    ⊢ Iff (Eq (LinearMap.range f) Bot.bot) (Eq f 0)
  -/
  rw [← range_le_bot_iff, le_bot_iff]
  /-
    🎉 no goals
  -/


theorem range_le_ker_iff {f : M →ₛₗ[τ₁₂] M₂} {g : M₂ →ₛₗ[τ₂₃] M₃} :
    range f ≤ ker g ↔ (g.comp f : M →ₛₗ[τ₁₃] M₃) = 0 :=
  ⟨fun h => ker_eq_top.1 <| eq_top_iff'.2 fun _ => h <| ⟨_, rfl⟩, fun h x hx =>
                                               /-
                                                 R : Type u_1
                                                 R₂ : Type u_2
                                                 R₃ : Type u_3
                                                 M : Type u_5
                                                 M₂ : Type u_6
                                                 M₃ : Type u_7
                                                 inst✝¹⁰ : Semiring R
                                                 inst✝⁹ : Semiring R₂
                                                 inst✝⁸ : Semiring R₃
                                                 inst✝⁷ : AddCommMonoid M
                                                 inst✝⁶ : AddCommMonoid M₂
                                                 inst✝⁵ : AddCommMonoid M₃
                                                 inst✝⁴ : Module R M
                                                 inst✝³ : Module R₂ M₂
                                                 inst✝² : Module R₃ M₃
                                                 τ₁₂ : RingHom R R₂
                                                 τ₂₃ : RingHom R₂ R₃
                                                 τ₁₃ : RingHom R R₃
                                                 inst✝¹ : RingHomCompTriple τ₁₂ τ₂₃ τ₁₃
                                                 inst✝ : RingHomSurjective τ₁₂
                                                 f : LinearMap τ₁₂ M M₂
                                                 g : LinearMap τ₂₃ M₂ M₃
                                                 h : Eq (g.comp f) 0
                                                 x : M₂
                                                 hx : Membership.mem (LinearMap.range f) x
                                                 y : M
                                                 hy : Eq (f y) x
                                                 ⊢ Eq (g x) 0
                                               -/
    mem_ker.2 <| Exists.elim hx fun y hy => by rw [← hy, ← comp_apply, h, zero_apply]⟩
                                               /-
                                                 🎉 no goals
                                               -/


theorem comap_le_comap_iff {f : F} (hf : range f = ⊤) {p p'} : comap f p ≤ comap f p' ↔ p ≤ p' :=
              /-
                R : Type u_1
                R₂ : Type u_2
                M : Type u_5
                M₂ : Type u_6
                inst✝⁸ : Semiring R
                inst✝⁷ : Semiring R₂
                inst✝⁶ : AddCommMonoid M
                inst✝⁵ : AddCommMonoid M₂
                inst✝⁴ : Module R M
                inst✝³ : Module R₂ M₂
                τ₁₂ : RingHom R R₂
                F : Type u_10
                inst✝² : FunLike F M M₂
                inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
                inst✝ : RingHomSurjective τ₁₂
                f : F
                hf : Eq (LinearMap.range f) Top.top
                p p' : Submodule R₂ M₂
                H : LE.le (Submodule.comap f p) (Submodule.comap f p')
                ⊢ LE.le p p'
              -/
  ⟨fun H ↦ by rwa [SetLike.le_def, (range_eq_top.1 hf).forall], comap_mono⟩
              /-
                🎉 no goals
              -/


theorem comap_injective {f : F} (hf : range f = ⊤) : Injective (comap f) := fun _ _ h =>
  le_antisymm ((comap_le_comap_iff hf).1 (le_of_eq h)) ((comap_le_comap_iff hf).1 (ge_of_eq h))

-- TODO (?): generalize to semilinear maps with `f ∘ₗ g` bijective.

theorem ker_eq_range_of_comp_eq_id {M P} [AddCommGroup M] [Module R M]
    [AddCommGroup P] [Module R P] {f : M →ₗ[R] P} {g : P →ₗ[R] M} (h : f ∘ₗ g = .id) :
    ker f = range (LinearMap.id - g ∘ₗ f) :=
                                                      /-
                                                        R : Type u_1
                                                        inst✝⁴ : Semiring R
                                                        M : Type u_11
                                                        P : Type u_12
                                                        inst✝³ : AddCommGroup M
                                                        inst✝² : Module R M
                                                        inst✝¹ : AddCommGroup P
                                                        inst✝ : Module R P
                                                        f : LinearMap (RingHom.id R) M P
                                                        g : LinearMap (RingHom.id R) P M
                                                        h : Eq (f.comp g) LinearMap.id
                                                        x : M
                                                        hx : Membership.mem (LinearMap.ker f) x
                                                        ⊢ Eq (HSub.hSub x (g (f x))) x
                                                      -/
  le_antisymm (fun x hx ↦ ⟨x, show x - g (f x) = x by rw [hx, map_zero, sub_zero]⟩) <|
                                                      /-
                                                        🎉 no goals
                                                      -/
                               /-
                                 R : Type u_1
                                 inst✝⁴ : Semiring R
                                 M : Type u_11
                                 P : Type u_12
                                 inst✝³ : AddCommGroup M
                                 inst✝² : Module R M
                                 inst✝¹ : AddCommGroup P
                                 inst✝ : Module R P
                                 f : LinearMap (RingHom.id R) M P
                                 g : LinearMap (RingHom.id R) P M
                                 h : Eq (f.comp g) LinearMap.id
                                 ⊢ Eq (f.comp (HSub.hSub LinearMap.id (g.comp f))) 0
                               -/
    range_le_ker_iff.mpr <| by rw [comp_sub, comp_id, ← comp_assoc, h, id_comp, sub_self]
                               /-
                                 🎉 no goals
                               -/


theorem range_toAddSubgroup [RingHomSurjective τ₁₂] (f : M →ₛₗ[τ₁₂] M₂) :
    (range f).toAddSubgroup = f.toAddMonoidHom.range :=
  rfl


theorem ker_le_iff [RingHomSurjective τ₁₂] {p : Submodule R M} :
    ker f ≤ p ↔ ∃ y ∈ range f, f ⁻¹' {y} ⊆ p := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_6
    inst✝⁸ : Ring R
    inst✝⁷ : Ring R₂
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : Module R M
    inst✝³ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    F : Type u_10
    inst✝² : FunLike F M M₂
    inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
    f : F
    inst✝ : RingHomSurjective τ₁₂
    p : Submodule R M
    ⊢ Iff (LE.le (LinearMap.ker f) p) (Exists fun y => And (Membership.mem (Linear …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_6
      inst✝⁸ : Ring R
      inst✝⁷ : Ring R₂
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M
      inst✝³ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      F : Type u_10
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
      f : F
      inst✝ : RingHomSurjective τ₁₂
      p : Submodule R M
      ⊢ LE.le (LinearMap.ker f) p → Exists fun y => And (Membership.mem (LinearMap.r …
    -/
  · intro h
    /-
      case mp
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_6
      inst✝⁸ : Ring R
      inst✝⁷ : Ring R₂
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M
      inst✝³ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      F : Type u_10
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
      f : F
      inst✝ : RingHomSurjective τ₁₂
      p : Submodule R M
      h : LE.le (LinearMap.ker f) p
      ⊢ Exists fun y => And (Membership.mem (LinearMap.range f) y) (HasSubset.Subset …
    -/
    use 0
    /-
      case h
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_6
      inst✝⁸ : Ring R
      inst✝⁷ : Ring R₂
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M
      inst✝³ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      F : Type u_10
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
      f : F
      inst✝ : RingHomSurjective τ₁₂
      p : Submodule R M
      h : LE.le (LinearMap.ker f) p
      ⊢ And (Membership.mem (LinearMap.range f) 0) (HasSubset.Subset (Set.preimage ( …
    -/
    rw [← SetLike.mem_coe, range_coe]
    /-
      case h
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_6
      inst✝⁸ : Ring R
      inst✝⁷ : Ring R₂
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M
      inst✝³ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      F : Type u_10
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
      f : F
      inst✝ : RingHomSurjective τ₁₂
      p : Submodule R M
      h : LE.le (LinearMap.ker f) p
      ⊢ And (Membership.mem (Set.range ⇑f) 0) (HasSubset.Subset (Set.preimage (⇑f) ( …
    -/
    exact ⟨⟨0, map_zero f⟩, h⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_6
      inst✝⁸ : Ring R
      inst✝⁷ : Ring R₂
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M
      inst✝³ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      F : Type u_10
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
      f : F
      inst✝ : RingHomSurjective τ₁₂
      p : Submodule R M
      ⊢ (Exists fun y => And (Membership.mem (LinearMap.range f) y) (HasSubset.Subse …
    -/
  · rintro ⟨y, h₁, h₂⟩
    /-
      case mpr.intro.intro
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_6
      inst✝⁸ : Ring R
      inst✝⁷ : Ring R₂
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M
      inst✝³ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      F : Type u_10
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
      f : F
      inst✝ : RingHomSurjective τ₁₂
      p : Submodule R M
      y : M₂
      h₁ : Membership.mem (LinearMap.range f) y
      h₂ : HasSubset.Subset (Set.preimage (⇑f) (Singleton.singleton y)) ↑p
      ⊢ LE.le (LinearMap.ker f) p
    -/
    rw [SetLike.le_def]
    /-
      case mpr.intro.intro
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_6
      inst✝⁸ : Ring R
      inst✝⁷ : Ring R₂
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M
      inst✝³ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      F : Type u_10
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
      f : F
      inst✝ : RingHomSurjective τ₁₂
      p : Submodule R M
      y : M₂
      h₁ : Membership.mem (LinearMap.range f) y
      h₂ : HasSubset.Subset (Set.preimage (⇑f) (Singleton.singleton y)) ↑p
      ⊢ ∀ ⦃x : M⦄, Membership.mem (LinearMap.ker f) x → Membership.mem p x
    -/
    intro z hz
    /-
      case mpr.intro.intro
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_6
      inst✝⁸ : Ring R
      inst✝⁷ : Ring R₂
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M
      inst✝³ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      F : Type u_10
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
      f : F
      inst✝ : RingHomSurjective τ₁₂
      p : Submodule R M
      y : M₂
      h₁ : Membership.mem (LinearMap.range f) y
      h₂ : HasSubset.Subset (Set.preimage (⇑f) (Singleton.singleton y)) ↑p
      z : M
      hz : Membership.mem (LinearMap.ker f) z
      ⊢ Membership.mem p z
    -/
    simp only [mem_ker, SetLike.mem_coe] at hz
    /-
      case mpr.intro.intro
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_6
      inst✝⁸ : Ring R
      inst✝⁷ : Ring R₂
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M
      inst✝³ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      F : Type u_10
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
      f : F
      inst✝ : RingHomSurjective τ₁₂
      p : Submodule R M
      y : M₂
      h₁ : Membership.mem (LinearMap.range f) y
      h₂ : HasSubset.Subset (Set.preimage (⇑f) (Singleton.singleton y)) ↑p
      z : M
      hz : Eq (f z) 0
      ⊢ Membership.mem p z
    -/
    rw [← SetLike.mem_coe, range_coe, Set.mem_range] at h₁
    /-
      case mpr.intro.intro
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_6
      inst✝⁸ : Ring R
      inst✝⁷ : Ring R₂
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M
      inst✝³ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      F : Type u_10
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
      f : F
      inst✝ : RingHomSurjective τ₁₂
      p : Submodule R M
      y : M₂
      h₁ : Exists fun y_1 => Eq (f y_1) y
      h₂ : HasSubset.Subset (Set.preimage (⇑f) (Singleton.singleton y)) ↑p
      z : M
      hz : Eq (f z) 0
      ⊢ Membership.mem p z
    -/
    obtain ⟨x, hx⟩ := h₁
    /-
      case mpr.intro.intro.intro
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_6
      inst✝⁸ : Ring R
      inst✝⁷ : Ring R₂
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M
      inst✝³ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      F : Type u_10
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
      f : F
      inst✝ : RingHomSurjective τ₁₂
      p : Submodule R M
      y : M₂
      h₂ : HasSubset.Subset (Set.preimage (⇑f) (Singleton.singleton y)) ↑p
      z : M
      hz : Eq (f z) 0
      x : M
      hx : Eq (f x) y
      ⊢ Membership.mem p z
    -/
    have hx' : x ∈ p := h₂ hx
    have hxz : z + x ∈ p := by
      apply h₂
      simp [hx, hz]
    /-
      case mpr.intro.intro.intro
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_6
      inst✝⁸ : Ring R
      inst✝⁷ : Ring R₂
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M
      inst✝³ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      F : Type u_10
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
      f : F
      inst✝ : RingHomSurjective τ₁₂
      p : Submodule R M
      y : M₂
      h₂ : HasSubset.Subset (Set.preimage (⇑f) (Singleton.singleton y)) ↑p
      z : M
      hz : Eq (f z) 0
      x : M
      hx : Eq (f x) y
      hx' : Membership.mem p x
      hxz : Membership.mem p (HAdd.hAdd z x)
      ⊢ Membership.mem p z
    -/
    suffices z + x - x ∈ p by simpa only [this, add_sub_cancel_right]
    /-
      case mpr.intro.intro.intro
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_6
      inst✝⁸ : Ring R
      inst✝⁷ : Ring R₂
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module R M
      inst✝³ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      F : Type u_10
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
      f : F
      inst✝ : RingHomSurjective τ₁₂
      p : Submodule R M
      y : M₂
      h₂ : HasSubset.Subset (Set.preimage (⇑f) (Singleton.singleton y)) ↑p
      z : M
      hz : Eq (f z) 0
      x : M
      hx : Eq (f x) y
      hx' : Membership.mem p x
      hxz : Membership.mem p (HAdd.hAdd z x)
      ⊢ Membership.mem p (HSub.hSub (HAdd.hAdd z x) x)
    -/
    exact p.sub_mem hxz hx'
    /-
      🎉 no goals
    -/


theorem range_smul (f : V →ₗ[K] V₂) (a : K) (h : a ≠ 0) : range (a • f) = range f := by
  /-
    K : Type u_4
    V : Type u_8
    V₂ : Type u_9
    inst✝⁴ : Semifield K
    inst✝³ : AddCommMonoid V
    inst✝² : Module K V
    inst✝¹ : AddCommMonoid V₂
    inst✝ : Module K V₂
    f : LinearMap (RingHom.id K) V V₂
    a : K
    h : Ne a 0
    ⊢ Eq (LinearMap.range (HSMul.hSMul a f)) (LinearMap.range f)
  -/
  simpa only [range_eq_map] using Submodule.map_smul f _ a h
  /-
    🎉 no goals
  -/


theorem range_smul' (f : V →ₗ[K] V₂) (a : K) :
    range (a • f) = ⨆ _ : a ≠ 0, range f := by
  /-
    K : Type u_4
    V : Type u_8
    V₂ : Type u_9
    inst✝⁴ : Semifield K
    inst✝³ : AddCommMonoid V
    inst✝² : Module K V
    inst✝¹ : AddCommMonoid V₂
    inst✝ : Module K V₂
    f : LinearMap (RingHom.id K) V V₂
    a : K
    ⊢ Eq (LinearMap.range (HSMul.hSMul a f)) (iSup fun x => LinearMap.range f)
  -/
  simpa only [range_eq_map] using Submodule.map_smul' f _ a
  /-
    🎉 no goals
  -/


@[simp]
theorem map_top [RingHomSurjective τ₁₂] (f : F) : map f ⊤ = range f :=
  (range_eq_map f).symm

@[simp]
                                                  /-
                                                    R : Type u_1
                                                    M : Type u_5
                                                    inst✝² : Semiring R
                                                    inst✝¹ : AddCommMonoid M
                                                    inst✝ : Module R M
                                                    p : Submodule R M
                                                    ⊢ Eq (LinearMap.range p.subtype) p
                                                  -/
theorem range_subtype : range p.subtype = p := by simpa using map_comap_subtype p ⊤
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem map_subtype_le (p' : Submodule R p) : map p.subtype p' ≤ p := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    p : Submodule R M
    p' : Submodule R (Subtype fun x => Membership.mem p x)
    ⊢ LE.le (Submodule.map p.subtype p') p
  -/
  simpa using (map_le_range : map p.subtype p' ≤ range p.subtype)
  /-
    🎉 no goals
  -/


/-- Under the canonical linear map from a submodule `p` to the ambient space `M`, the image of the
maximal submodule of `p` is just `p`. -/
                                                                      /-
                                                                        R : Type u_1
                                                                        M : Type u_5
                                                                        inst✝² : Semiring R
                                                                        inst✝¹ : AddCommMonoid M
                                                                        inst✝ : Module R M
                                                                        p : Submodule R M
                                                                        ⊢ Eq (Submodule.map p.subtype Top.top) p
                                                                      -/
theorem map_subtype_top : map p.subtype (⊤ : Submodule R p) = p := by simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem comap_subtype_eq_top {p p' : Submodule R M} : comap p.subtype p' = ⊤ ↔ p ≤ p' :=
                                                           /-
                                                             R : Type u_1
                                                             M : Type u_5
                                                             inst✝² : Semiring R
                                                             inst✝¹ : AddCommMonoid M
                                                             inst✝ : Module R M
                                                             p p' : Submodule R M
                                                             ⊢ Iff (LE.le (Submodule.map p.subtype Top.top) p') (LE.le p p')
                                                           -/
  eq_top_iff.trans <| map_le_iff_le_comap.symm.trans <| by rw [map_subtype_top]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
theorem comap_subtype_self : comap p.subtype p = ⊤ :=
  comap_subtype_eq_top.2 le_rfl


@[simp]
lemma comap_subtype_le_iff {p q r : Submodule R M} :
    q.comap p.subtype ≤ r.comap p.subtype ↔ p ⊓ q ≤ p ⊓ r :=
              /-
                R : Type u_1
                M : Type u_5
                inst✝² : Semiring R
                inst✝¹ : AddCommMonoid M
                inst✝ : Module R M
                p q r : Submodule R M
                h : LE.le (Submodule.comap p.subtype q) (Submodule.comap p.subtype r)
                ⊢ LE.le (Min.min p q) (Min.min p r)
              -/
  ⟨fun h ↦ by simpa using map_mono (f := p.subtype) h,
              /-
                🎉 no goals
              -/
              /-
                R : Type u_1
                M : Type u_5
                inst✝² : Semiring R
                inst✝¹ : AddCommMonoid M
                inst✝ : Module R M
                p q r : Submodule R M
                h : LE.le (Min.min p q) (Min.min p r)
                ⊢ LE.le (Submodule.comap p.subtype q) (Submodule.comap p.subtype r)
              -/
   fun h ↦ by simpa using comap_mono (f := p.subtype) h⟩
              /-
                🎉 no goals
              -/


theorem range_inclusion (p q : Submodule R M) (h : p ≤ q) :
    range (inclusion h) = comap q.subtype p := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    p q : Submodule R M
    h : LE.le p q
    ⊢ Eq (LinearMap.range (Submodule.inclusion h)) (Submodule.comap q.subtype p)
  -/
  rw [← map_top, inclusion, LinearMap.map_codRestrict, map_top, range_subtype]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_subtype_range_inclusion {p p' : Submodule R M} (h : p ≤ p') :
                                                    /-
                                                      R : Type u_1
                                                      M : Type u_5
                                                      inst✝² : Semiring R
                                                      inst✝¹ : AddCommMonoid M
                                                      inst✝ : Module R M
                                                      p p' : Submodule R M
                                                      h : LE.le p p'
                                                      ⊢ Eq (Submodule.map p'.subtype (LinearMap.range (Submodule.inclusion h))) p
                                                    -/
    map p'.subtype (range <| inclusion h) = p := by simp [range_inclusion, map_comap_eq, h]
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- If `N ⊆ M` then submodules of `N` are the same as submodules of `M` contained in `N`.

See also `Submodule.mapIic`. -/
def MapSubtype.relIso : Submodule R p ≃o { p' : Submodule R M // p' ≤ p } where
  toFun p' := ⟨map p.subtype p', map_subtype_le p _⟩
  invFun q := comap p.subtype q
                                               /-
                                                 R : Type u_1
                                                 R₂ : Type u_2
                                                 R₃ : Type u_3
                                                 K : Type u_4
                                                 M : Type u_5
                                                 M₂ : Type u_6
                                                 M₃ : Type u_7
                                                 V : Type u_8
                                                 V₂ : Type u_9
                                                 inst✝⁷ : Semiring R
                                                 inst✝⁶ : Semiring R₂
                                                 inst✝⁵ : AddCommMonoid M
                                                 inst✝⁴ : AddCommMonoid M₂
                                                 inst✝³ : Module R M
                                                 inst✝² : Module R₂ M₂
                                                 p : Submodule R M
                                                 τ₁₂ : RingHom R R₂
                                                 F : Type u_10
                                                 inst✝¹ : FunLike F M M₂
                                                 inst✝ : SemilinearMapClass F τ₁₂ M M₂
                                                 p' : Submodule R (Subtype fun x => Membership.mem p x)
                                                 ⊢ Function.Injective ⇑p.subtype
                                               -/
  left_inv p' := comap_map_eq_of_injective (by exact Subtype.val_injective) p'
                                               /-
                                                 🎉 no goals
                                               -/
                                                    /-
                                                      R : Type u_1
                                                      R₂ : Type u_2
                                                      R₃ : Type u_3
                                                      K : Type u_4
                                                      M : Type u_5
                                                      M₂ : Type u_6
                                                      M₃ : Type u_7
                                                      V : Type u_8
                                                      V₂ : Type u_9
                                                      inst✝⁷ : Semiring R
                                                      inst✝⁶ : Semiring R₂
                                                      inst✝⁵ : AddCommMonoid M
                                                      inst✝⁴ : AddCommMonoid M₂
                                                      inst✝³ : Module R M
                                                      inst✝² : Module R₂ M₂
                                                      p : Submodule R M
                                                      τ₁₂ : RingHom R R₂
                                                      F : Type u_10
                                                      inst✝¹ : FunLike F M M₂
                                                      inst✝ : SemilinearMapClass F τ₁₂ M M₂
                                                      x✝ : Subtype fun p' => LE.le p' p
                                                      q : Submodule R M
                                                      hq : LE.le q p
                                                      ⊢ Eq ↑((fun p' => ⟨Submodule.map p.subtype p', ⋯⟩) ((fun q => Submodule.comap  …
                                                    -/
  right_inv := fun ⟨q, hq⟩ => Subtype.ext_val <| by simp [map_comap_subtype p, inf_of_le_right hq]
                                                    /-
                                                      🎉 no goals
                                                    -/
  map_rel_iff' {p₁ p₂} := Subtype.coe_le_coe.symm.trans <| by
    /-
      R : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      K : Type u_4
      M : Type u_5
      M₂ : Type u_6
      M₃ : Type u_7
      V : Type u_8
      V₂ : Type u_9
      inst✝⁷ : Semiring R
      inst✝⁶ : Semiring R₂
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : Module R M
      inst✝² : Module R₂ M₂
      p : Submodule R M
      τ₁₂ : RingHom R R₂
      F : Type u_10
      inst✝¹ : FunLike F M M₂
      inst✝ : SemilinearMapClass F τ₁₂ M M₂
      p₁ p₂ : Submodule R (Subtype fun x => Membership.mem p x)
      ⊢ Iff (LE.le ↑({ toFun := fun p' => ⟨Submodule.map p.subtype p', ⋯⟩, invFun := …
    -/
    dsimp
    rw [map_le_iff_le_comap,
      comap_map_eq_of_injective (show Injective p.subtype from Subtype.coe_injective) p₂]


/-- If `p ⊆ M` is a submodule, the ordering of submodules of `p` is embedded in the ordering of
submodules of `M`. -/
def MapSubtype.orderEmbedding : Submodule R p ↪o Submodule R M :=
  (RelIso.toRelEmbedding <| MapSubtype.relIso p).trans <|
    Subtype.relEmbedding (X := Submodule R M) (fun p p' ↦ p ≤ p') _


@[simp]
theorem map_subtype_embedding_eq (p' : Submodule R p) :
    MapSubtype.orderEmbedding p p' = map p.subtype p' :=
  rfl


/-- If `N ⊆ M` then submodules of `N` are the same as submodules of `M` contained in `N`. -/
def mapIic (p : Submodule R M) :
    Submodule R p ≃o Set.Iic p :=
  Submodule.MapSubtype.relIso p


@[simp] lemma coe_mapIic_apply
    (p : Submodule R M) (q : Submodule R p) :
    (p.mapIic q : Submodule R M) = q.map p.subtype :=
  rfl


/-- A monomorphism is injective. -/
theorem ker_eq_bot_of_cancel {f : M →ₛₗ[τ₁₂] M₂}
    (h : ∀ u v : ker f →ₗ[R] M, f.comp u = f.comp v → u = v) : ker f = ⊥ := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_6
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring R₂
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    f : LinearMap τ₁₂ M M₂
    h : ∀ (u v : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (Linear …
    ⊢ Eq (LinearMap.ker f) Bot.bot
  -/
  have h₁ : f.comp (0 : ker f →ₗ[R] M) = 0 := comp_zero _
  rw [← Submodule.range_subtype (ker f),
    ← h 0 f.ker.subtype (Eq.trans h₁ (comp_ker_subtype f).symm)]
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_6
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring R₂
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    f : LinearMap τ₁₂ M M₂
    h : ∀ (u v : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (Linear …
    h₁ : Eq (f.comp 0) 0
    ⊢ Eq (LinearMap.range 0) Bot.bot
  -/
  exact range_zero
  /-
    🎉 no goals
  -/


theorem range_comp_of_range_eq_top [RingHomSurjective τ₁₂] [RingHomSurjective τ₂₃]
    [RingHomSurjective τ₁₃] {f : M →ₛₗ[τ₁₂] M₂} (g : M₂ →ₛₗ[τ₂₃] M₃) (hf : range f = ⊤) :
                                                     /-
                                                       R : Type u_1
                                                       R₂ : Type u_2
                                                       R₃ : Type u_3
                                                       M : Type u_5
                                                       M₂ : Type u_6
                                                       M₃ : Type u_7
                                                       inst✝¹² : Semiring R
                                                       inst✝¹¹ : Semiring R₂
                                                       inst✝¹⁰ : Semiring R₃
                                                       inst✝⁹ : AddCommMonoid M
                                                       inst✝⁸ : AddCommMonoid M₂
                                                       inst✝⁷ : AddCommMonoid M₃
                                                       inst✝⁶ : Module R M
                                                       inst✝⁵ : Module R₂ M₂
                                                       inst✝⁴ : Module R₃ M₃
                                                       τ₁₂ : RingHom R R₂
                                                       τ₂₃ : RingHom R₂ R₃
                                                       τ₁₃ : RingHom R R₃
                                                       inst✝³ : RingHomCompTriple τ₁₂ τ₂₃ τ₁₃
                                                       inst✝² : RingHomSurjective τ₁₂
                                                       inst✝¹ : RingHomSurjective τ₂₃
                                                       inst✝ : RingHomSurjective τ₁₃
                                                       f : LinearMap τ₁₂ M M₂
                                                       g : LinearMap τ₂₃ M₂ M₃
                                                       hf : Eq (LinearMap.range f) Top.top
                                                       ⊢ Eq (LinearMap.range (g.comp f)) (LinearMap.range g)
                                                     -/
    range (g.comp f : M →ₛₗ[τ₁₃] M₃) = range g := by rw [range_comp, hf, Submodule.map_top]
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- If `O` is a submodule of `M`, and `Φ : O →ₗ M'` is a linear map,
then `(ϕ : O →ₗ M').submoduleImage N` is `ϕ(N)` as a submodule of `M'` -/
def submoduleImage {M' : Type*} [AddCommMonoid M'] [Module R M'] {O : Submodule R M}
    (ϕ : O →ₗ[R] M') (N : Submodule R M) : Submodule R M' :=
  (N.comap O.subtype).map ϕ


@[simp]
theorem mem_submoduleImage {M' : Type*} [AddCommMonoid M'] [Module R M'] {O : Submodule R M}
    {ϕ : O →ₗ[R] M'} {N : Submodule R M} {x : M'} :
    x ∈ ϕ.submoduleImage N ↔ ∃ (y : _) (yO : y ∈ O), y ∈ N ∧ ϕ ⟨y, yO⟩ = x := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    M' : Type u_10
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    O : Submodule R M
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) M'
    N : Submodule R M
    x : M'
    ⊢ Iff (Membership.mem (ϕ.submoduleImage N) x) (Exists fun y => Exists fun yO = …
  -/
  refine Submodule.mem_map.trans ⟨?_, ?_⟩ <;> simp_rw [Submodule.mem_comap]
    /-
      case refine_1
      R : Type u_1
      M : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type u_10
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      O : Submodule R M
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) M'
      N : Submodule R M
      x : M'
      ⊢ (Exists fun y => And (Membership.mem N (O.subtype y)) (Eq (ϕ y) x)) → Exists …
    -/
  · rintro ⟨⟨y, yO⟩, yN : y ∈ N, h⟩
    /-
      case refine_1.intro.mk.intro
      R : Type u_1
      M : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type u_10
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      O : Submodule R M
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) M'
      N : Submodule R M
      x : M'
      y : M
      yO : Membership.mem O y
      yN : Membership.mem N y
      h : Eq (ϕ ⟨y, yO⟩) x
      ⊢ Exists fun y => Exists fun yO => And (Membership.mem N y) (Eq (ϕ ⟨y, yO⟩) x)
    -/
    exact ⟨y, yO, yN, h⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      M : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type u_10
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      O : Submodule R M
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) M'
      N : Submodule R M
      x : M'
      ⊢ (Exists fun y => Exists fun yO => And (Membership.mem N y) (Eq (ϕ ⟨y, yO⟩) x …
    -/
  · rintro ⟨y, yO, yN, h⟩
    /-
      case refine_2.intro.intro.intro
      R : Type u_1
      M : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type u_10
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      O : Submodule R M
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) M'
      N : Submodule R M
      x : M'
      y : M
      yO : Membership.mem O y
      yN : Membership.mem N y
      h : Eq (ϕ ⟨y, yO⟩) x
      ⊢ Exists fun y => And (Membership.mem N (O.subtype y)) (Eq (ϕ y) x)
    -/
    exact ⟨⟨y, yO⟩, yN, h⟩
    /-
      🎉 no goals
    -/


theorem mem_submoduleImage_of_le {M' : Type*} [AddCommMonoid M'] [Module R M'] {O : Submodule R M}
    {ϕ : O →ₗ[R] M'} {N : Submodule R M} (hNO : N ≤ O) {x : M'} :
    x ∈ ϕ.submoduleImage N ↔ ∃ (y : _) (yN : y ∈ N), ϕ ⟨y, hNO yN⟩ = x := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    M' : Type u_10
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    O : Submodule R M
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) M'
    N : Submodule R M
    hNO : LE.le N O
    x : M'
    ⊢ Iff (Membership.mem (ϕ.submoduleImage N) x) (Exists fun y => Exists fun yN = …
  -/
  refine mem_submoduleImage.trans ⟨?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      M : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type u_10
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      O : Submodule R M
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) M'
      N : Submodule R M
      hNO : LE.le N O
      x : M'
      ⊢ (Exists fun y => Exists fun yO => And (Membership.mem N y) (Eq (ϕ ⟨y, yO⟩) x …
    -/
  · rintro ⟨y, yO, yN, h⟩
    /-
      case refine_1.intro.intro.intro
      R : Type u_1
      M : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type u_10
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      O : Submodule R M
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) M'
      N : Submodule R M
      hNO : LE.le N O
      x : M'
      y : M
      yO : Membership.mem O y
      yN : Membership.mem N y
      h : Eq (ϕ ⟨y, yO⟩) x
      ⊢ Exists fun y => Exists fun yN => Eq (ϕ ⟨y, ⋯⟩) x
    -/
    exact ⟨y, yN, h⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      M : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type u_10
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      O : Submodule R M
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) M'
      N : Submodule R M
      hNO : LE.le N O
      x : M'
      ⊢ (Exists fun y => Exists fun yN => Eq (ϕ ⟨y, ⋯⟩) x) → Exists fun y => Exists  …
    -/
  · rintro ⟨y, yN, h⟩
    /-
      case refine_2.intro.intro
      R : Type u_1
      M : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type u_10
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      O : Submodule R M
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) M'
      N : Submodule R M
      hNO : LE.le N O
      x : M'
      y : M
      yN : Membership.mem N y
      h : Eq (ϕ ⟨y, ⋯⟩) x
      ⊢ Exists fun y => Exists fun yO => And (Membership.mem N y) (Eq (ϕ ⟨y, yO⟩) x)
    -/
    exact ⟨y, hNO yN, yN, h⟩
    /-
      🎉 no goals
    -/


theorem submoduleImage_apply_of_le {M' : Type*} [AddCommGroup M'] [Module R M']
    {O : Submodule R M} (ϕ : O →ₗ[R] M') (N : Submodule R M) (hNO : N ≤ O) :
    ϕ.submoduleImage N = range (ϕ.comp (Submodule.inclusion hNO)) := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    M' : Type u_10
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    O : Submodule R M
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) M'
    N : Submodule R M
    hNO : LE.le N O
    ⊢ Eq (ϕ.submoduleImage N) (LinearMap.range (ϕ.comp (Submodule.inclusion hNO)))
  -/
  rw [submoduleImage, range_comp, Submodule.range_inclusion]
  /-
    🎉 no goals
  -/


                                                                      /-
                                                                        R : Type u_1
                                                                        R₂ : Type u_2
                                                                        M : Type u_5
                                                                        M₂ : Type u_6
                                                                        inst✝⁶ : Semiring R
                                                                        inst✝⁵ : Semiring R₂
                                                                        inst✝⁴ : AddCommMonoid M
                                                                        inst✝³ : AddCommMonoid M₂
                                                                        inst✝² : Module R M
                                                                        inst✝¹ : Module R₂ M₂
                                                                        τ₁₂ : RingHom R R₂
                                                                        inst✝ : RingHomSurjective τ₁₂
                                                                        f : LinearMap τ₁₂ M M₂
                                                                        ⊢ Eq (LinearMap.range f.rangeRestrict) Top.top
                                                                      -/
@[simp] theorem range_rangeRestrict : range f.rangeRestrict = ⊤ := by simp [f.range_codRestrict _]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem surjective_rangeRestrict : Surjective f.rangeRestrict := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_6
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring R₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    ⊢ Function.Surjective ⇑f.rangeRestrict
  -/
  rw [← range_eq_top, range_rangeRestrict]
  /-
    🎉 no goals
  -/


@[simp] theorem ker_rangeRestrict : ker f.rangeRestrict = ker f := LinearMap.ker_codRestrict _ _ _


@[simp] theorem injective_rangeRestrict_iff : Injective f.rangeRestrict ↔ Injective f :=
  Set.injective_codRestrict _


