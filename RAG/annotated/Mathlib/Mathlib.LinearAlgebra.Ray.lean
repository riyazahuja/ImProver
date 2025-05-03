/-- Two vectors are in the same ray if either one of them is zero or some positive multiples of them
are equal (in the typical case over a field, this means one of them is a nonnegative multiple of
the other). -/
def SameRay (v₁ v₂ : M) : Prop :=
  v₁ = 0 ∨ v₂ = 0 ∨ ∃ r₁ r₂ : R, 0 < r₁ ∧ 0 < r₂ ∧ r₁ • v₁ = r₂ • v₂


@[simp]
theorem zero_left (y : M) : SameRay R 0 y :=
  Or.inl rfl


@[simp]
theorem zero_right (x : M) : SameRay R x 0 :=
  Or.inr <| Or.inl rfl


@[nontriviality]
theorem of_subsingleton [Subsingleton M] (x y : M) : SameRay R x y := by
  /-
    R : Type u_1
    inst✝³ : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Subsingleton M
    x y : M
    ⊢ SameRay R x y
  -/
  rw [Subsingleton.elim x 0]
  /-
    R : Type u_1
    inst✝³ : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Subsingleton M
    x y : M
    ⊢ SameRay R 0 y
  -/
  exact zero_left _
  /-
    🎉 no goals
  -/


@[nontriviality]
theorem of_subsingleton' [Subsingleton R] (x y : M) : SameRay R x y :=
  haveI := Module.subsingleton R M
  of_subsingleton x y


/-- `SameRay` is reflexive. -/
@[refl]
theorem refl (x : M) : SameRay R x x := by
  /-
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x : M
    ⊢ SameRay R x x
  -/
  nontriviality R
  /-
    R : Type u_1
    inst✝³ : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    x : M
    inst✝ : Nontrivial R
    ⊢ SameRay R x x
  -/
  exact Or.inr (Or.inr <| ⟨1, 1, zero_lt_one, zero_lt_one, rfl⟩)
  /-
    🎉 no goals
  -/


protected theorem rfl : SameRay R x x :=
  refl _


/-- `SameRay` is symmetric. -/
@[symm]
theorem symm (h : SameRay R x y) : SameRay R y x :=
  (or_left_comm.1 h).imp_right <| Or.imp_right fun ⟨r₁, r₂, h₁, h₂, h⟩ => ⟨r₂, r₁, h₂, h₁, h.symm⟩


/-- If `x` and `y` are nonzero vectors on the same ray, then there exist positive numbers `r₁ r₂`
such that `r₁ • x = r₂ • y`. -/
theorem exists_pos (h : SameRay R x y) (hx : x ≠ 0) (hy : y ≠ 0) :
    ∃ r₁ r₂ : R, 0 < r₁ ∧ 0 < r₂ ∧ r₁ • x = r₂ • y :=
  (h.resolve_left hx).resolve_left hy


theorem sameRay_comm : SameRay R x y ↔ SameRay R y x :=
  ⟨SameRay.symm, SameRay.symm⟩


/-- `SameRay` is transitive unless the vector in the middle is zero and both other vectors are
nonzero. -/
theorem trans (hxy : SameRay R x y) (hyz : SameRay R y z) (hy : y = 0 → x = 0 ∨ z = 0) :
    SameRay R x z := by
  /-
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x y z : M
    hxy : SameRay R x y
    hyz : SameRay R y z
    hy : Eq y 0 → Or (Eq x 0) (Eq z 0)
    ⊢ SameRay R x z
  -/
  rcases eq_or_ne x 0 with (rfl | hx); · exact zero_left z
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x y z : M
    hxy : SameRay R x y
    hyz : SameRay R y z
    hy : Eq y 0 → Or (Eq x 0) (Eq z 0)
    hx : Ne x 0
    ⊢ SameRay R x z
  -/
  rcases eq_or_ne z 0 with (rfl | hz); · exact zero_right x
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr.inr
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x y z : M
    hxy : SameRay R x y
    hyz : SameRay R y z
    hy : Eq y 0 → Or (Eq x 0) (Eq z 0)
    hx : Ne x 0
    hz : Ne z 0
    ⊢ SameRay R x z
  -/
  rcases eq_or_ne y 0 with (rfl | hy)
    /-
      case inr.inr.inl
      R : Type u_1
      inst✝² : StrictOrderedCommSemiring R
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x z : M
      hx : Ne x 0
      hz : Ne z 0
      hxy : SameRay R x 0
      hyz : SameRay R 0 z
      hy : Eq 0 0 → Or (Eq x 0) (Eq z 0)
      ⊢ SameRay R x z
    -/
  · exact (hy rfl).elim (fun h => (hx h).elim) fun h => (hz h).elim
    /-
      🎉 no goals
    -/
  /-
    case inr.inr.inr
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x y z : M
    hxy : SameRay R x y
    hyz : SameRay R y z
    hy✝ : Eq y 0 → Or (Eq x 0) (Eq z 0)
    hx : Ne x 0
    hz : Ne z 0
    hy : Ne y 0
    ⊢ SameRay R x z
  -/
  rcases hxy.exists_pos hx hy with ⟨r₁, r₂, hr₁, hr₂, h₁⟩
  /-
    case inr.inr.inr.intro.intro.intro.intro
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x y z : M
    hxy : SameRay R x y
    hyz : SameRay R y z
    hy✝ : Eq y 0 → Or (Eq x 0) (Eq z 0)
    hx : Ne x 0
    hz : Ne z 0
    hy : Ne y 0
    r₁ r₂ : R
    hr₁ : LT.lt 0 r₁
    hr₂ : LT.lt 0 r₂
    h₁ : Eq (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ y)
    ⊢ SameRay R x z
  -/
  rcases hyz.exists_pos hy hz with ⟨r₃, r₄, hr₃, hr₄, h₂⟩
  /-
    case inr.inr.inr.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x y z : M
    hxy : SameRay R x y
    hyz : SameRay R y z
    hy✝ : Eq y 0 → Or (Eq x 0) (Eq z 0)
    hx : Ne x 0
    hz : Ne z 0
    hy : Ne y 0
    r₁ r₂ : R
    hr₁ : LT.lt 0 r₁
    hr₂ : LT.lt 0 r₂
    h₁ : Eq (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ y)
    r₃ r₄ : R
    hr₃ : LT.lt 0 r₃
    hr₄ : LT.lt 0 r₄
    h₂ : Eq (HSMul.hSMul r₃ y) (HSMul.hSMul r₄ z)
    ⊢ SameRay R x z
  -/
  refine Or.inr (Or.inr <| ⟨r₃ * r₁, r₂ * r₄, mul_pos hr₃ hr₁, mul_pos hr₂ hr₄, ?_⟩)
  /-
    case inr.inr.inr.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x y z : M
    hxy : SameRay R x y
    hyz : SameRay R y z
    hy✝ : Eq y 0 → Or (Eq x 0) (Eq z 0)
    hx : Ne x 0
    hz : Ne z 0
    hy : Ne y 0
    r₁ r₂ : R
    hr₁ : LT.lt 0 r₁
    hr₂ : LT.lt 0 r₂
    h₁ : Eq (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ y)
    r₃ r₄ : R
    hr₃ : LT.lt 0 r₃
    hr₄ : LT.lt 0 r₄
    h₂ : Eq (HSMul.hSMul r₃ y) (HSMul.hSMul r₄ z)
    ⊢ Eq (HSMul.hSMul (HMul.hMul r₃ r₁) x) (HSMul.hSMul (HMul.hMul r₂ r₄) z)
  -/
  rw [mul_smul, mul_smul, h₁, ← h₂, smul_comm]
  /-
    🎉 no goals
  -/


/-- A vector is in the same ray as a nonnegative multiple of itself. -/
lemma sameRay_nonneg_smul_right (v : M) (h : 0 ≤ a) : SameRay R v (a • v) := by
  /-
    R : Type u_1
    inst✝⁷ : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    S : Type u_5
    inst✝⁴ : OrderedCommSemiring S
    inst✝³ : Algebra S R
    inst✝² : Module S M
    inst✝¹ : SMulPosMono S R
    inst✝ : IsScalarTower S R M
    a : S
    v : M
    h : LE.le 0 a
    ⊢ SameRay R v (HSMul.hSMul a v)
  -/
  obtain h | h := (algebraMap_nonneg R h).eq_or_gt
    /-
      case inl
      R : Type u_1
      inst✝⁷ : StrictOrderedCommSemiring R
      M : Type u_2
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      S : Type u_5
      inst✝⁴ : OrderedCommSemiring S
      inst✝³ : Algebra S R
      inst✝² : Module S M
      inst✝¹ : SMulPosMono S R
      inst✝ : IsScalarTower S R M
      a : S
      v : M
      h✝ : LE.le 0 a
      h : Eq ((algebraMap S R) a) 0
      ⊢ SameRay R v (HSMul.hSMul a v)
    -/
  · rw [← algebraMap_smul R a v, h, zero_smul]
    /-
      case inl
      R : Type u_1
      inst✝⁷ : StrictOrderedCommSemiring R
      M : Type u_2
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      S : Type u_5
      inst✝⁴ : OrderedCommSemiring S
      inst✝³ : Algebra S R
      inst✝² : Module S M
      inst✝¹ : SMulPosMono S R
      inst✝ : IsScalarTower S R M
      a : S
      v : M
      h✝ : LE.le 0 a
      h : Eq ((algebraMap S R) a) 0
      ⊢ SameRay R v 0
    -/
    exact zero_right _
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝⁷ : StrictOrderedCommSemiring R
      M : Type u_2
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      S : Type u_5
      inst✝⁴ : OrderedCommSemiring S
      inst✝³ : Algebra S R
      inst✝² : Module S M
      inst✝¹ : SMulPosMono S R
      inst✝ : IsScalarTower S R M
      a : S
      v : M
      h✝ : LE.le 0 a
      h : LT.lt 0 ((algebraMap S R) a)
      ⊢ SameRay R v (HSMul.hSMul a v)
    -/
  · refine Or.inr <| Or.inr ⟨algebraMap S R a, 1, h, by nontriviality R; exact zero_lt_one, ?_⟩
    /-
      case inr
      R : Type u_1
      inst✝⁷ : StrictOrderedCommSemiring R
      M : Type u_2
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      S : Type u_5
      inst✝⁴ : OrderedCommSemiring S
      inst✝³ : Algebra S R
      inst✝² : Module S M
      inst✝¹ : SMulPosMono S R
      inst✝ : IsScalarTower S R M
      a : S
      v : M
      h✝ : LE.le 0 a
      h : LT.lt 0 ((algebraMap S R) a)
      ⊢ Eq (HSMul.hSMul ((algebraMap S R) a) v) (HSMul.hSMul 1 (HSMul.hSMul a v))
    -/
    module
    /-
      🎉 no goals
    -/


/-- A nonnegative multiple of a vector is in the same ray as that vector. -/
lemma sameRay_nonneg_smul_left (v : M) (ha : 0 ≤ a) : SameRay R (a • v) v :=
  (sameRay_nonneg_smul_right v ha).symm


/-- A vector is in the same ray as a positive multiple of itself. -/
lemma sameRay_pos_smul_right (v : M) (ha : 0 < a) : SameRay R v (a • v) :=
  sameRay_nonneg_smul_right v ha.le


/-- A positive multiple of a vector is in the same ray as that vector. -/
lemma sameRay_pos_smul_left (v : M) (ha : 0 < a) : SameRay R (a • v) v :=
  sameRay_nonneg_smul_left v ha.le


/-- A vector is in the same ray as a nonnegative multiple of one it is in the same ray as. -/
lemma nonneg_smul_right (h : SameRay R x y) (ha : 0 ≤ a) : SameRay R x (a • y) :=
                                                                  /-
                                                                    R : Type u_1
                                                                    inst✝⁷ : StrictOrderedCommSemiring R
                                                                    M : Type u_2
                                                                    inst✝⁶ : AddCommMonoid M
                                                                    inst✝⁵ : Module R M
                                                                    x y : M
                                                                    S : Type u_5
                                                                    inst✝⁴ : OrderedCommSemiring S
                                                                    inst✝³ : Algebra S R
                                                                    inst✝² : Module S M
                                                                    inst✝¹ : SMulPosMono S R
                                                                    inst✝ : IsScalarTower S R M
                                                                    a : S
                                                                    h : SameRay R x y
                                                                    ha : LE.le 0 a
                                                                    hy : Eq y 0
                                                                    ⊢ Eq (HSMul.hSMul a y) 0
                                                                  -/
  h.trans (sameRay_nonneg_smul_right y ha) fun hy => Or.inr <| by rw [hy, smul_zero]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- A nonnegative multiple of a vector is in the same ray as one it is in the same ray as. -/
lemma nonneg_smul_left (h : SameRay R x y) (ha : 0 ≤ a) : SameRay R (a • x) y :=
  (h.symm.nonneg_smul_right ha).symm


/-- A vector is in the same ray as a positive multiple of one it is in the same ray as. -/
theorem pos_smul_right (h : SameRay R x y) (ha : 0 < a) : SameRay R x (a • y) :=
  h.nonneg_smul_right ha.le


/-- A positive multiple of a vector is in the same ray as one it is in the same ray as. -/
theorem pos_smul_left (h : SameRay R x y) (hr : 0 < a) : SameRay R (a • x) y :=
  h.nonneg_smul_left hr.le


/-- If two vectors are on the same ray then they remain so after applying a linear map. -/
theorem map (f : M →ₗ[R] N) (h : SameRay R x y) : SameRay R (f x) (f y) :=
                      /-
                        R : Type u_1
                        inst✝⁴ : StrictOrderedCommSemiring R
                        M : Type u_2
                        inst✝³ : AddCommMonoid M
                        inst✝² : Module R M
                        N : Type u_3
                        inst✝¹ : AddCommMonoid N
                        inst✝ : Module R N
                        x y : M
                        f : LinearMap (RingHom.id R) M N
                        h : SameRay R x y
                        hx : Eq x 0
                        ⊢ Eq (f x) 0
                      -/
  (h.imp fun hx => by rw [hx, map_zero]) <|
                      /-
                        🎉 no goals
                      -/
                         /-
                           R : Type u_1
                           inst✝⁴ : StrictOrderedCommSemiring R
                           M : Type u_2
                           inst✝³ : AddCommMonoid M
                           inst✝² : Module R M
                           N : Type u_3
                           inst✝¹ : AddCommMonoid N
                           inst✝ : Module R N
                           x y : M
                           f : LinearMap (RingHom.id R) M N
                           h : SameRay R x y
                           hy : Eq y 0
                           ⊢ Eq (f y) 0
                         -/
    Or.imp (fun hy => by rw [hy, map_zero]) fun ⟨r₁, r₂, hr₁, hr₂, h⟩ =>
                         /-
                           🎉 no goals
                         -/
                            /-
                              R : Type u_1
                              inst✝⁴ : StrictOrderedCommSemiring R
                              M : Type u_2
                              inst✝³ : AddCommMonoid M
                              inst✝² : Module R M
                              N : Type u_3
                              inst✝¹ : AddCommMonoid N
                              inst✝ : Module R N
                              x y : M
                              f : LinearMap (RingHom.id R) M N
                              h✝ : SameRay R x y
                              x✝ : Exists fun r₁ => Exists fun r₂ => And (LT.lt 0 r₁) (And (LT.lt 0 r₂) (Eq  …
                              r₁ r₂ : R
                              hr₁ : LT.lt 0 r₁
                              hr₂ : LT.lt 0 r₂
                              h : Eq (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ y)
                              ⊢ Eq (HSMul.hSMul r₁ (f x)) (HSMul.hSMul r₂ (f y))
                            -/
      ⟨r₁, r₂, hr₁, hr₂, by rw [← f.map_smul, ← f.map_smul, h]⟩
                            /-
                              🎉 no goals
                            -/


/-- The images of two vectors under an injective linear map are on the same ray if and only if the
original vectors are on the same ray. -/
theorem _root_.Function.Injective.sameRay_map_iff
    {F : Type*} [FunLike F M N] [LinearMapClass F R M N]
    {f : F} (hf : Function.Injective f) :
    SameRay R (f x) (f y) ↔ SameRay R x y := by
  /-
    R : Type u_1
    inst✝⁶ : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    N : Type u_3
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    x y : M
    F : Type u_6
    inst✝¹ : FunLike F M N
    inst✝ : LinearMapClass F R M N
    f : F
    hf : Function.Injective ⇑f
    ⊢ Iff (SameRay R (f x) (f y)) (SameRay R x y)
  -/
  simp only [SameRay, map_zero, ← hf.eq_iff, map_smul]
  /-
    🎉 no goals
  -/


/-- The images of two vectors under a linear equivalence are on the same ray if and only if the
original vectors are on the same ray. -/
@[simp]
theorem sameRay_map_iff (e : M ≃ₗ[R] N) : SameRay R (e x) (e y) ↔ SameRay R x y :=
  Function.Injective.sameRay_map_iff (EquivLike.injective e)


/-- If two vectors are on the same ray then both scaled by the same action are also on the same
ray. -/
theorem smul {S : Type*} [Monoid S] [DistribMulAction S M] [SMulCommClass R S M]
    (h : SameRay R x y) (s : S) : SameRay R (s • x) (s • y) :=
  h.map (s • (LinearMap.id : M →ₗ[R] M))


/-- If `x` and `y` are on the same ray as `z`, then so is `x + y`. -/
theorem add_left (hx : SameRay R x z) (hy : SameRay R y z) : SameRay R (x + y) z := by
  /-
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x y z : M
    hx : SameRay R x z
    hy : SameRay R y z
    ⊢ SameRay R (HAdd.hAdd x y) z
  -/
  rcases eq_or_ne x 0 with (rfl | hx₀); · rwa [zero_add]
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case inr
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x y z : M
    hx : SameRay R x z
    hy : SameRay R y z
    hx₀ : Ne x 0
    ⊢ SameRay R (HAdd.hAdd x y) z
  -/
  rcases eq_or_ne y 0 with (rfl | hy₀); · rwa [add_zero]
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case inr.inr
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x y z : M
    hx : SameRay R x z
    hy : SameRay R y z
    hx₀ : Ne x 0
    hy₀ : Ne y 0
    ⊢ SameRay R (HAdd.hAdd x y) z
  -/
  rcases eq_or_ne z 0 with (rfl | hz₀); · apply zero_right
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case inr.inr.inr
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x y z : M
    hx : SameRay R x z
    hy : SameRay R y z
    hx₀ : Ne x 0
    hy₀ : Ne y 0
    hz₀ : Ne z 0
    ⊢ SameRay R (HAdd.hAdd x y) z
  -/
  rcases hx.exists_pos hx₀ hz₀ with ⟨rx, rz₁, hrx, hrz₁, Hx⟩
  /-
    case inr.inr.inr.intro.intro.intro.intro
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x y z : M
    hx : SameRay R x z
    hy : SameRay R y z
    hx₀ : Ne x 0
    hy₀ : Ne y 0
    hz₀ : Ne z 0
    rx rz₁ : R
    hrx : LT.lt 0 rx
    hrz₁ : LT.lt 0 rz₁
    Hx : Eq (HSMul.hSMul rx x) (HSMul.hSMul rz₁ z)
    ⊢ SameRay R (HAdd.hAdd x y) z
  -/
  rcases hy.exists_pos hy₀ hz₀ with ⟨ry, rz₂, hry, hrz₂, Hy⟩
  /-
    case inr.inr.inr.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x y z : M
    hx : SameRay R x z
    hy : SameRay R y z
    hx₀ : Ne x 0
    hy₀ : Ne y 0
    hz₀ : Ne z 0
    rx rz₁ : R
    hrx : LT.lt 0 rx
    hrz₁ : LT.lt 0 rz₁
    Hx : Eq (HSMul.hSMul rx x) (HSMul.hSMul rz₁ z)
    ry rz₂ : R
    hry : LT.lt 0 ry
    hrz₂ : LT.lt 0 rz₂
    Hy : Eq (HSMul.hSMul ry y) (HSMul.hSMul rz₂ z)
    ⊢ SameRay R (HAdd.hAdd x y) z
  -/
  refine Or.inr (Or.inr ⟨rx * ry, ry * rz₁ + rx * rz₂, mul_pos hrx hry, ?_, ?_⟩)
    /-
      case inr.inr.inr.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      R : Type u_1
      inst✝² : StrictOrderedCommSemiring R
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x y z : M
      hx : SameRay R x z
      hy : SameRay R y z
      hx₀ : Ne x 0
      hy₀ : Ne y 0
      hz₀ : Ne z 0
      rx rz₁ : R
      hrx : LT.lt 0 rx
      hrz₁ : LT.lt 0 rz₁
      Hx : Eq (HSMul.hSMul rx x) (HSMul.hSMul rz₁ z)
      ry rz₂ : R
      hry : LT.lt 0 ry
      hrz₂ : LT.lt 0 rz₂
      Hy : Eq (HSMul.hSMul ry y) (HSMul.hSMul rz₂ z)
      ⊢ LT.lt 0 (HAdd.hAdd (HMul.hMul ry rz₁) (HMul.hMul rx rz₂))
    -/
  · positivity
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.inr.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
      R : Type u_1
      inst✝² : StrictOrderedCommSemiring R
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x y z : M
      hx : SameRay R x z
      hy : SameRay R y z
      hx₀ : Ne x 0
      hy₀ : Ne y 0
      hz₀ : Ne z 0
      rx rz₁ : R
      hrx : LT.lt 0 rx
      hrz₁ : LT.lt 0 rz₁
      Hx : Eq (HSMul.hSMul rx x) (HSMul.hSMul rz₁ z)
      ry rz₂ : R
      hry : LT.lt 0 ry
      hrz₂ : LT.lt 0 rz₂
      Hy : Eq (HSMul.hSMul ry y) (HSMul.hSMul rz₂ z)
      ⊢ Eq (HSMul.hSMul (HMul.hMul rx ry) (HAdd.hAdd x y)) (HSMul.hSMul (HAdd.hAdd ( …
    -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  · convert congr(ry • $Hx + rx • $Hy) using 1 <;> module
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- If `y` and `z` are on the same ray as `x`, then so is `y + z`. -/
theorem add_right (hy : SameRay R x y) (hz : SameRay R x z) : SameRay R x (y + z) :=
  (hy.symm.add_left hz.symm).symm


set_option linter.unusedVariables false in
/-- Nonzero vectors, as used to define rays. This type depends on an unused argument `R` so that
`RayVector.Setoid` can be an instance. -/
@[nolint unusedArguments]
def RayVector (R M : Type*) [Zero M] :=
  { v : M // v ≠ 0 }


instance RayVector.coe [Zero M] : CoeOut (RayVector R M) M where
  coe := Subtype.val


instance {R M : Type*} [Zero M] [Nontrivial M] : Nonempty (RayVector R M) :=
  let ⟨x, hx⟩ := exists_ne (0 : M)
  ⟨⟨x, hx⟩⟩

/-- The setoid of the `SameRay` relation for the subtype of nonzero vectors. -/
instance RayVector.Setoid : Setoid (RayVector R M) where
  r x y := SameRay R (x : M) y
  iseqv :=
    ⟨fun _ => SameRay.refl _, fun h => h.symm, by
      /-
        R : Type u_1
        inst✝⁵ : StrictOrderedCommSemiring R
        M : Type u_2
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        N : Type u_3
        inst✝² : AddCommMonoid N
        inst✝¹ : Module R N
        ι : Type u_4
        inst✝ : DecidableEq ι
        ⊢ ∀ {x y z : RayVector R M}, SameRay R ↑x ↑y → SameRay R ↑y ↑z → SameRay R ↑x ↑z
      -/
      intros x y z hxy hyz
      /-
        R : Type u_1
        inst✝⁵ : StrictOrderedCommSemiring R
        M : Type u_2
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        N : Type u_3
        inst✝² : AddCommMonoid N
        inst✝¹ : Module R N
        ι : Type u_4
        inst✝ : DecidableEq ι
        x y z : RayVector R M
        hxy : SameRay R ↑x ↑y
        hyz : SameRay R ↑y ↑z
        ⊢ SameRay R ↑x ↑z
      -/
      exact hxy.trans hyz fun hy => (y.2 hy).elim⟩
      /-
        🎉 no goals
      -/


/-- A ray (equivalence class of nonzero vectors with common positive multiples) in a module. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed has_nonempty_instance nolint, no such linter
def Module.Ray :=
  Quotient (RayVector.Setoid R M)


/-- Equivalence of nonzero vectors, in terms of `SameRay`. -/
theorem equiv_iff_sameRay {v₁ v₂ : RayVector R M} : v₁ ≈ v₂ ↔ SameRay R (v₁ : M) v₂ :=
  Iff.rfl


/-- The ray given by a nonzero vector. -/
def rayOfNeZero (v : M) (h : v ≠ 0) : Module.Ray R M :=
  ⟦⟨v, h⟩⟧


/-- An induction principle for `Module.Ray`, used as `induction x using Module.Ray.ind`. -/
theorem Module.Ray.ind {C : Module.Ray R M → Prop} (h : ∀ (v) (hv : v ≠ 0), C (rayOfNeZero R v hv))
    (x : Module.Ray R M) : C x :=
  Quotient.ind (Subtype.rec <| h) x


instance [Nontrivial M] : Nonempty (Module.Ray R M) :=
  Nonempty.map Quotient.mk' inferInstance


/-- The rays given by two nonzero vectors are equal if and only if those vectors
satisfy `SameRay`. -/
theorem ray_eq_iff {v₁ v₂ : M} (hv₁ : v₁ ≠ 0) (hv₂ : v₂ ≠ 0) :
    rayOfNeZero R _ hv₁ = rayOfNeZero R _ hv₂ ↔ SameRay R v₁ v₂ :=
  Quotient.eq'


/-- The ray given by a positive multiple of a nonzero vector. -/
@[simp]
theorem ray_pos_smul {v : M} (h : v ≠ 0) {r : R} (hr : 0 < r) (hrv : r • v ≠ 0) :
    rayOfNeZero R (r • v) hrv = rayOfNeZero R v h :=
  (ray_eq_iff _ _).2 <| SameRay.sameRay_pos_smul_left v hr


/-- An equivalence between modules implies an equivalence between ray vectors. -/
def RayVector.mapLinearEquiv (e : M ≃ₗ[R] N) : RayVector R M ≃ RayVector R N :=
  Equiv.subtypeEquiv e.toEquiv fun _ => e.map_ne_zero_iff.symm


/-- An equivalence between modules implies an equivalence between rays. -/
def Module.Ray.map (e : M ≃ₗ[R] N) : Module.Ray R M ≃ Module.Ray R N :=
  Quotient.congr (RayVector.mapLinearEquiv e) fun _ _=> (SameRay.sameRay_map_iff _).symm


@[simp]
theorem Module.Ray.map_apply (e : M ≃ₗ[R] N) (v : M) (hv : v ≠ 0) :
    Module.Ray.map e (rayOfNeZero _ v hv) = rayOfNeZero _ (e v) (e.map_ne_zero_iff.2 hv) :=
  rfl


@[simp]
theorem Module.Ray.map_refl : (Module.Ray.map <| LinearEquiv.refl R M) = Equiv.refl _ :=
  Equiv.ext <| Module.Ray.ind R fun _ _ => rfl


@[simp]
theorem Module.Ray.map_symm (e : M ≃ₗ[R] N) : (Module.Ray.map e).symm = Module.Ray.map e.symm :=
  rfl


/-- Any invertible action preserves the non-zeroness of ray vectors. This is primarily of interest
when `G = Rˣ` -/
instance {R : Type*} : MulAction G (RayVector R M) where
  smul r := Subtype.map (r • ·) fun _ => (smul_ne_zero_iff_ne _).2
  mul_smul a b _ := Subtype.ext <| mul_smul a b _
  one_smul _ := Subtype.ext <| one_smul _ _


/-- Any invertible action preserves the non-zeroness of rays. This is primarily of interest when
`G = Rˣ` -/
instance : MulAction G (Module.Ray R M) where
  smul r := Quotient.map (r • ·) fun _ _ h => h.smul _
  mul_smul a b := Quotient.ind fun _ => congr_arg Quotient.mk' <| mul_smul a b _
  one_smul := Quotient.ind fun _ => congr_arg Quotient.mk' <| one_smul _ _


/-- The action via `LinearEquiv.apply_distribMulAction` corresponds to `Module.Ray.map`. -/
@[simp]
theorem Module.Ray.linearEquiv_smul_eq_map (e : M ≃ₗ[R] M) (v : Module.Ray R M) :
    e • v = Module.Ray.map e v :=
  rfl


@[simp]
theorem smul_rayOfNeZero (g : G) (v : M) (hv) :
    g • rayOfNeZero R v hv = rayOfNeZero R (g • v) ((smul_ne_zero_iff_ne _).2 hv) :=
  rfl


/-- Scaling by a positive unit is a no-op. -/
theorem units_smul_of_pos (u : Rˣ) (hu : 0 < (u.1 : R)) (v : Module.Ray R M) : u • v = v := by
  /-
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    u : Units R
    hu : LT.lt 0 ↑u
    v : Module.Ray R M
    ⊢ Eq (HSMul.hSMul u v) v
  -/
  induction v using Module.Ray.ind
  /-
    case h
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    u : Units R
    hu : LT.lt 0 ↑u
    v✝ : M
    hv✝ : Ne v✝ 0
    ⊢ Eq (HSMul.hSMul u (rayOfNeZero R v✝ hv✝)) (rayOfNeZero R v✝ hv✝)
  -/
  rw [smul_rayOfNeZero, ray_eq_iff]
  /-
    case h
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    u : Units R
    hu : LT.lt 0 ↑u
    v✝ : M
    hv✝ : Ne v✝ 0
    ⊢ SameRay R (HSMul.hSMul u v✝) v✝
  -/
  exact SameRay.sameRay_pos_smul_left _ hu
  /-
    🎉 no goals
  -/


/-- An arbitrary `RayVector` giving a ray. -/
def someRayVector (x : Module.Ray R M) : RayVector R M :=
  Quotient.out x


/-- The ray of `someRayVector`. -/
@[simp]
theorem someRayVector_ray (x : Module.Ray R M) : (⟦x.someRayVector⟧ : Module.Ray R M) = x :=
  Quotient.out_eq _


/-- An arbitrary nonzero vector giving a ray. -/
def someVector (x : Module.Ray R M) : M :=
  x.someRayVector


/-- `someVector` is nonzero. -/
@[simp]
theorem someVector_ne_zero (x : Module.Ray R M) : x.someVector ≠ 0 :=
  x.someRayVector.property


/-- The ray of `someVector`. -/
@[simp]
theorem someVector_ray (x : Module.Ray R M) : rayOfNeZero R _ x.someVector_ne_zero = x :=
  (congr_arg _ (Subtype.coe_eta _ _) : _).trans x.out_eq


/-- `SameRay.neg` as an `iff`. -/
@[simp]
theorem sameRay_neg_iff : SameRay R (-x) (-y) ↔ SameRay R x y := by
  /-
    R : Type u_1
    inst✝² : StrictOrderedCommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    ⊢ Iff (SameRay R (Neg.neg x) (Neg.neg y)) (SameRay R x y)
  -/
  simp only [SameRay, neg_eq_zero, smul_neg, neg_inj]
  /-
    🎉 no goals
  -/


alias ⟨SameRay.of_neg, SameRay.neg⟩ := sameRay_neg_iff


                                                                     /-
                                                                       R : Type u_1
                                                                       inst✝² : StrictOrderedCommRing R
                                                                       M : Type u_2
                                                                       inst✝¹ : AddCommGroup M
                                                                       inst✝ : Module R M
                                                                       x y : M
                                                                       ⊢ Iff (SameRay R (Neg.neg x) y) (SameRay R x (Neg.neg y))
                                                                     -/
theorem sameRay_neg_swap : SameRay R (-x) y ↔ SameRay R x (-y) := by rw [← sameRay_neg_iff, neg_neg]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem eq_zero_of_sameRay_neg_smul_right [NoZeroSMulDivisors R M] {r : R} (hr : r < 0)
    (h : SameRay R x (r • x)) : x = 0 := by
  /-
    R : Type u_1
    inst✝³ : StrictOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    x : M
    inst✝ : NoZeroSMulDivisors R M
    r : R
    hr : LT.lt r 0
    h : SameRay R x (HSMul.hSMul r x)
    ⊢ Eq x 0
  -/
  rcases h with (rfl | h₀ | ⟨r₁, r₂, hr₁, hr₂, h⟩)
    /-
      case inl
      R : Type u_1
      inst✝³ : StrictOrderedCommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : NoZeroSMulDivisors R M
      r : R
      hr : LT.lt r 0
      ⊢ Eq 0 0
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      R : Type u_1
      inst✝³ : StrictOrderedCommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      x : M
      inst✝ : NoZeroSMulDivisors R M
      r : R
      hr : LT.lt r 0
      h₀ : Eq (HSMul.hSMul r x) 0
      ⊢ Eq x 0
    -/
  · simpa [hr.ne] using h₀
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.intro.intro.intro.intro
      R : Type u_1
      inst✝³ : StrictOrderedCommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      x : M
      inst✝ : NoZeroSMulDivisors R M
      r : R
      hr : LT.lt r 0
      r₁ r₂ : R
      hr₁ : LT.lt 0 r₁
      hr₂ : LT.lt 0 r₂
      h : Eq (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ (HSMul.hSMul r x))
      ⊢ Eq x 0
    -/
  · rw [← sub_eq_zero, smul_smul, ← sub_smul, smul_eq_zero] at h
    /-
      case inr.inr.intro.intro.intro.intro
      R : Type u_1
      inst✝³ : StrictOrderedCommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      x : M
      inst✝ : NoZeroSMulDivisors R M
      r : R
      hr : LT.lt r 0
      r₁ r₂ : R
      hr₁ : LT.lt 0 r₁
      hr₂ : LT.lt 0 r₂
      h : Or (Eq (HSub.hSub r₁ (HMul.hMul r₂ r)) 0) (Eq x 0)
      ⊢ Eq x 0
    -/
    refine h.resolve_left (ne_of_gt <| sub_pos.2 ?_)
    /-
      case inr.inr.intro.intro.intro.intro
      R : Type u_1
      inst✝³ : StrictOrderedCommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      x : M
      inst✝ : NoZeroSMulDivisors R M
      r : R
      hr : LT.lt r 0
      r₁ r₂ : R
      hr₁ : LT.lt 0 r₁
      hr₂ : LT.lt 0 r₂
      h : Or (Eq (HSub.hSub r₁ (HMul.hMul r₂ r)) 0) (Eq x 0)
      ⊢ LT.lt (HMul.hMul r₂ r) r₁
    -/
    exact (mul_neg_of_pos_of_neg hr₂ hr).trans hr₁
    /-
      🎉 no goals
    -/


/-- If a vector is in the same ray as its negation, that vector is zero. -/
theorem eq_zero_of_sameRay_self_neg [NoZeroSMulDivisors R M] (h : SameRay R x (-x)) : x = 0 := by
  /-
    R : Type u_1
    inst✝³ : StrictOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    x : M
    inst✝ : NoZeroSMulDivisors R M
    h : SameRay R x (Neg.neg x)
    ⊢ Eq x 0
  -/
  nontriviality M; haveI : Nontrivial R := Module.nontrivial R M
  /-
    R : Type u_1
    inst✝³ : StrictOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    x : M
    inst✝ : NoZeroSMulDivisors R M
    h : SameRay R x (Neg.neg x)
    a✝ : Nontrivial M
    this : Nontrivial R
    ⊢ Eq x 0
  -/
  refine eq_zero_of_sameRay_neg_smul_right (neg_lt_zero.2 (zero_lt_one' R)) ?_
  /-
    R : Type u_1
    inst✝³ : StrictOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    x : M
    inst✝ : NoZeroSMulDivisors R M
    h : SameRay R x (Neg.neg x)
    a✝ : Nontrivial M
    this : Nontrivial R
    ⊢ SameRay R x (HSMul.hSMul (-1) x)
  -/
  rwa [neg_one_smul]
  /-
    🎉 no goals
  -/


/-- Negating a nonzero vector. -/
instance {R : Type*} : Neg (RayVector R M) :=
  ⟨fun v => ⟨-v, neg_ne_zero.2 v.prop⟩⟩


/-- Negating a nonzero vector commutes with coercion to the underlying module. -/
@[simp, norm_cast]
theorem coe_neg {R : Type*} (v : RayVector R M) : ↑(-v) = -(v : M) :=
  rfl


/-- Negating a nonzero vector twice produces the original vector. -/
instance {R : Type*} : InvolutiveNeg (RayVector R M) where
  neg := Neg.neg
                  /-
                    R✝ : Type u_1
                    inst✝⁴ : StrictOrderedCommRing R✝
                    M : Type u_2
                    N : Type u_3
                    inst✝³ : AddCommGroup M
                    inst✝² : AddCommGroup N
                    inst✝¹ : Module R✝ M
                    inst✝ : Module R✝ N
                    x y : M
                    R : Type u_4
                    v : RayVector R M
                    ⊢ Eq (Neg.neg (Neg.neg v)) v
                  -/
  neg_neg v := by rw [Subtype.ext_iff, coe_neg, coe_neg, neg_neg]
                  /-
                    🎉 no goals
                  -/


/-- If two nonzero vectors are equivalent, so are their negations. -/
@[simp]
theorem equiv_neg_iff {v₁ v₂ : RayVector R M} : -v₁ ≈ -v₂ ↔ v₁ ≈ v₂ :=
  sameRay_neg_iff


/-- Negating a ray. -/
instance : Neg (Module.Ray R M) :=
  ⟨Quotient.map (fun v => -v) fun _ _ => RayVector.equiv_neg_iff.2⟩


/-- The ray given by the negation of a nonzero vector. -/
@[simp]
theorem neg_rayOfNeZero (v : M) (h : v ≠ 0) :
    -rayOfNeZero R _ h = rayOfNeZero R (-v) (neg_ne_zero.2 h) :=
  rfl


/-- Negating a ray twice produces the original ray. -/
instance : InvolutiveNeg (Module.Ray R M) where
  neg := Neg.neg
                  /-
                    R : Type u_1
                    inst✝⁴ : StrictOrderedCommRing R
                    M : Type u_2
                    N : Type u_3
                    inst✝³ : AddCommGroup M
                    inst✝² : AddCommGroup N
                    inst✝¹ : Module R M
                    inst✝ : Module R N
                    x✝ y : M
                    x : Module.Ray R M
                    ⊢ Eq (Neg.neg (Neg.neg x)) x
                  -/
  neg_neg x := by apply ind R (by simp) x
                  /-
                    🎉 no goals
                  -/
  -- Quotient.ind (fun a => congr_arg Quotient.mk' <| neg_neg _) x


/-- A ray does not equal its own negation. -/
theorem ne_neg_self [NoZeroSMulDivisors R M] (x : Module.Ray R M) : x ≠ -x := by
  /-
    R : Type u_1
    inst✝³ : StrictOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x : Module.Ray R M
    ⊢ Ne x (Neg.neg x)
  -/
  induction' x using Module.Ray.ind with x hx
  /-
    case h
    R : Type u_1
    inst✝³ : StrictOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x : M
    hx : Ne x 0
    ⊢ Ne (rayOfNeZero R x hx) (Neg.neg (rayOfNeZero R x hx))
  -/
  rw [neg_rayOfNeZero, Ne, ray_eq_iff]
  /-
    case h
    R : Type u_1
    inst✝³ : StrictOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x : M
    hx : Ne x 0
    ⊢ Not (SameRay R x (Neg.neg x))
  -/
  exact mt eq_zero_of_sameRay_self_neg hx
  /-
    🎉 no goals
  -/


theorem neg_units_smul (u : Rˣ) (v : Module.Ray R M) : -u • v = -(u • v) := by
  /-
    R : Type u_1
    inst✝² : StrictOrderedCommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    u : Units R
    v : Module.Ray R M
    ⊢ Eq (HSMul.hSMul (Neg.neg u) v) (Neg.neg (HSMul.hSMul u v))
  -/
  induction v using Module.Ray.ind
  /-
    case h
    R : Type u_1
    inst✝² : StrictOrderedCommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    u : Units R
    v✝ : M
    hv✝ : Ne v✝ 0
    ⊢ Eq (HSMul.hSMul (Neg.neg u) (rayOfNeZero R v✝ hv✝)) (Neg.neg (HSMul.hSMul u  …
  -/
  simp only [smul_rayOfNeZero, Units.smul_def, Units.val_neg, neg_smul, neg_rayOfNeZero]
  /-
    🎉 no goals
  -/

-- Porting note: `(u.1 : R)` was `(u : R)`, CoeHead from R to Rˣ does not seem to work.

/-- Scaling by a negative unit is negation. -/
theorem units_smul_of_neg (u : Rˣ) (hu : u.1 < 0) (v : Module.Ray R M) : u • v = -v := by
  /-
    R : Type u_1
    inst✝² : StrictOrderedCommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    u : Units R
    hu : LT.lt (↑u) 0
    v : Module.Ray R M
    ⊢ Eq (HSMul.hSMul u v) (Neg.neg v)
  -/
  rw [← neg_inj, neg_neg, ← neg_units_smul, units_smul_of_pos]
  /-
    case hu
    R : Type u_1
    inst✝² : StrictOrderedCommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    u : Units R
    hu : LT.lt (↑u) 0
    v : Module.Ray R M
    ⊢ LT.lt 0 ↑(Neg.neg u)
  -/
  rwa [Units.val_neg, Right.neg_pos_iff]
  /-
    🎉 no goals
  -/


@[simp]
protected theorem map_neg (f : M ≃ₗ[R] N) (v : Module.Ray R M) : map f (-v) = -map f v := by
  /-
    R : Type u_1
    inst✝⁴ : StrictOrderedCommRing R
    M : Type u_2
    N : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : LinearEquiv (RingHom.id R) M N
    v : Module.Ray R M
    ⊢ Eq ((Module.Ray.map f) (Neg.neg v)) (Neg.neg ((Module.Ray.map f) v))
  -/
  induction' v using Module.Ray.ind with g hg
  /-
    case h
    R : Type u_1
    inst✝⁴ : StrictOrderedCommRing R
    M : Type u_2
    N : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : LinearEquiv (RingHom.id R) M N
    g : M
    hg : Ne g 0
    ⊢ Eq ((Module.Ray.map f) (Neg.neg (rayOfNeZero R g hg))) (Neg.neg ((Module.Ray …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- `SameRay` follows from membership of `MulAction.orbit` for the `Units.posSubgroup`. -/
theorem sameRay_of_mem_orbit {v₁ v₂ : M} (h : v₁ ∈ MulAction.orbit ↥(Units.posSubgroup R) v₂) :
    SameRay R v₁ v₂ := by
  /-
    R : Type u_1
    inst✝² : LinearOrderedCommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    v₁ v₂ : M
    h : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Units.po …
    ⊢ SameRay R v₁ v₂
  -/
  rcases h with ⟨⟨r, hr : 0 < r.1⟩, rfl : r • v₂ = v₁⟩
  /-
    case intro.mk
    R : Type u_1
    inst✝² : LinearOrderedCommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    v₂ : M
    r : Units R
    hr : LT.lt 0 ↑r
    ⊢ SameRay R (HSMul.hSMul r v₂) v₂
  -/
  exact SameRay.sameRay_pos_smul_left _ hr
  /-
    🎉 no goals
  -/


/-- Scaling by an inverse unit is the same as scaling by itself. -/
@[simp]
theorem units_inv_smul (u : Rˣ) (v : Module.Ray R M) : u⁻¹ • v = u • v :=
  have := mul_self_pos.2 u.ne_zero
  calc
                                                                                /-
                                                                                  R : Type u_1
                                                                                  inst✝² : LinearOrderedCommRing R
                                                                                  M : Type u_2
                                                                                  inst✝¹ : AddCommGroup M
                                                                                  inst✝ : Module R M
                                                                                  u : Units R
                                                                                  v : Module.Ray R M
                                                                                  this : LT.lt 0 (HMul.hMul ↑u ↑u)
                                                                                  ⊢ LT.lt 0 ↑(HMul.hMul u u)
                                                                                -/
    u⁻¹ • v = (u * u) • u⁻¹ • v := Eq.symm <| (u⁻¹ • v).units_smul_of_pos _ (by exact this)
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
                    /-
                      R : Type u_1
                      inst✝² : LinearOrderedCommRing R
                      M : Type u_2
                      inst✝¹ : AddCommGroup M
                      inst✝ : Module R M
                      u : Units R
                      v : Module.Ray R M
                      this : LT.lt 0 (HMul.hMul ↑u ↑u)
                      ⊢ Eq (HSMul.hSMul (HMul.hMul u u) (HSMul.hSMul (Inv.inv u) v)) (HSMul.hSMul u v)
                    -/
    _ = u • v := by rw [mul_smul, smul_inv_smul]
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem sameRay_smul_right_iff {v : M} {r : R} : SameRay R v (r • v) ↔ 0 ≤ r ∨ v = 0 :=
  ⟨fun hrv => or_iff_not_imp_left.2 fun hr => eq_zero_of_sameRay_neg_smul_right (not_le.1 hr) hrv,
    or_imp.2 ⟨SameRay.sameRay_nonneg_smul_right v, fun h => h.symm ▸ SameRay.zero_left _⟩⟩


/-- A nonzero vector is in the same ray as a multiple of itself if and only if that multiple
is positive. -/
theorem sameRay_smul_right_iff_of_ne {v : M} (hv : v ≠ 0) {r : R} (hr : r ≠ 0) :
    SameRay R v (r • v) ↔ 0 < r := by
  /-
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    v : M
    hv : Ne v 0
    r : R
    hr : Ne r 0
    ⊢ Iff (SameRay R v (HSMul.hSMul r v)) (LT.lt 0 r)
  -/
  simp only [sameRay_smul_right_iff, hv, or_false, hr.symm.le_iff_lt]
  /-
    🎉 no goals
  -/


@[simp]
theorem sameRay_smul_left_iff {v : M} {r : R} : SameRay R (r • v) v ↔ 0 ≤ r ∨ v = 0 :=
  SameRay.sameRay_comm.trans sameRay_smul_right_iff


/-- A multiple of a nonzero vector is in the same ray as that vector if and only if that multiple
is positive. -/
theorem sameRay_smul_left_iff_of_ne {v : M} (hv : v ≠ 0) {r : R} (hr : r ≠ 0) :
    SameRay R (r • v) v ↔ 0 < r :=
  SameRay.sameRay_comm.trans (sameRay_smul_right_iff_of_ne hv hr)


@[simp]
theorem sameRay_neg_smul_right_iff {v : M} {r : R} : SameRay R (-v) (r • v) ↔ r ≤ 0 ∨ v = 0 := by
  /-
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    v : M
    r : R
    ⊢ Iff (SameRay R (Neg.neg v) (HSMul.hSMul r v)) (Or (LE.le r 0) (Eq v 0))
  -/
  rw [← sameRay_neg_iff, neg_neg, ← neg_smul, sameRay_smul_right_iff, neg_nonneg]
  /-
    🎉 no goals
  -/


theorem sameRay_neg_smul_right_iff_of_ne {v : M} {r : R} (hv : v ≠ 0) (hr : r ≠ 0) :
    SameRay R (-v) (r • v) ↔ r < 0 := by
  /-
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    v : M
    r : R
    hv : Ne v 0
    hr : Ne r 0
    ⊢ Iff (SameRay R (Neg.neg v) (HSMul.hSMul r v)) (LT.lt r 0)
  -/
  simp only [sameRay_neg_smul_right_iff, hv, or_false, hr.le_iff_lt]
  /-
    🎉 no goals
  -/


@[simp]
theorem sameRay_neg_smul_left_iff {v : M} {r : R} : SameRay R (r • v) (-v) ↔ r ≤ 0 ∨ v = 0 :=
  SameRay.sameRay_comm.trans sameRay_neg_smul_right_iff


theorem sameRay_neg_smul_left_iff_of_ne {v : M} {r : R} (hv : v ≠ 0) (hr : r ≠ 0) :
    SameRay R (r • v) (-v) ↔ r < 0 :=
  SameRay.sameRay_comm.trans <| sameRay_neg_smul_right_iff_of_ne hv hr

-- Porting note: `(u.1 : R)` was `(u : R)`, CoeHead from R to Rˣ does not seem to work.

@[simp]
theorem units_smul_eq_self_iff {u : Rˣ} {v : Module.Ray R M} : u • v = v ↔ 0 < u.1 := by
  /-
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    u : Units R
    v : Module.Ray R M
    ⊢ Iff (Eq (HSMul.hSMul u v) v) (LT.lt 0 ↑u)
  -/
  induction' v using Module.Ray.ind with v hv
  /-
    case h
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    u : Units R
    v : M
    hv : Ne v 0
    ⊢ Iff (Eq (HSMul.hSMul u (rayOfNeZero R v hv)) (rayOfNeZero R v hv)) (LT.lt 0  …
  -/
  simp only [smul_rayOfNeZero, ray_eq_iff, Units.smul_def, sameRay_smul_left_iff_of_ne hv u.ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem units_smul_eq_neg_iff {u : Rˣ} {v : Module.Ray R M} : u • v = -v ↔ u.1 < 0 := by
  rw [← neg_inj, neg_neg, ← Module.Ray.neg_units_smul, units_smul_eq_self_iff, Units.val_neg,
    neg_pos]


/-- Two vectors are in the same ray, or the first is in the same ray as the negation of the
second, if and only if they are not linearly independent. -/
theorem sameRay_or_sameRay_neg_iff_not_linearIndependent {x y : M} :
    SameRay R x y ∨ SameRay R x (-y) ↔ ¬LinearIndependent R ![x, y] := by
  /-
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x y : M
    ⊢ Iff (Or (SameRay R x y) (SameRay R x (Neg.neg y))) (Not (LinearIndependent R …
  -/
  by_cases hx : x = 0; · simpa [hx] using fun h : LinearIndependent R ![0, y] => h.ne_zero 0 rfl
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x y : M
    hx : Not (Eq x 0)
    ⊢ Iff (Or (SameRay R x y) (SameRay R x (Neg.neg y))) (Not (LinearIndependent R …
  -/
  by_cases hy : y = 0; · simpa [hy] using fun h : LinearIndependent R ![x, 0] => h.ne_zero 1 rfl
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x y : M
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Iff (Or (SameRay R x y) (SameRay R x (Neg.neg y))) (Not (LinearIndependent R …
  -/
  simp_rw [Fintype.not_linearIndependent_iff]
  /-
    case neg
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x y : M
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Iff (Or (SameRay R x y) (SameRay R x (Neg.neg y))) (Exists fun g => And (Eq  …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case neg.refine_1
      R : Type u_1
      inst✝³ : LinearOrderedCommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : NoZeroSMulDivisors R M
      x y : M
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      h : Or (SameRay R x y) (SameRay R x (Neg.neg y))
      ⊢ Exists fun g => And (Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (Matrix. …
    -/
  · rcases h with ((hx0 | hy0 | ⟨r₁, r₂, hr₁, _, h⟩) | (hx0 | hy0 | ⟨r₁, r₂, hr₁, _, h⟩))
      /-
        case neg.refine_1.inl.inl
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        hx0 : Eq x 0
        ⊢ Exists fun g => And (Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (Matrix. …
      -/
    · exact False.elim (hx hx0)
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_1.inl.inr.inl
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        hy0 : Eq y 0
        ⊢ Exists fun g => And (Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (Matrix. …
      -/
    · exact False.elim (hy hy0)
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_1.inl.inr.inr.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        r₁ r₂ : R
        hr₁ : LT.lt 0 r₁
        left✝ : LT.lt 0 r₂
        h : Eq (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ y)
        ⊢ Exists fun g => And (Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (Matrix. …
      -/
    · refine ⟨![r₁, -r₂], ?_⟩
      /-
        case neg.refine_1.inl.inr.inr.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        r₁ r₂ : R
        hr₁ : LT.lt 0 r₁
        left✝ : LT.lt 0 r₂
        h : Eq (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ y)
        ⊢ And (Eq (Finset.univ.sum fun i => HSMul.hSMul (Matrix.vecCons r₁ (Matrix.vec …
      -/
      rw [Fin.sum_univ_two, Fin.exists_fin_two]
      /-
        case neg.refine_1.inl.inr.inr.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        r₁ r₂ : R
        hr₁ : LT.lt 0 r₁
        left✝ : LT.lt 0 r₂
        h : Eq (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ y)
        ⊢ And (Eq (HAdd.hAdd (HSMul.hSMul (Matrix.vecCons r₁ (Matrix.vecCons (Neg.neg  …
      -/
      simp [h, hr₁.ne.symm]
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_1.inr.inl
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        hx0 : Eq x 0
        ⊢ Exists fun g => And (Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (Matrix. …
      -/
    · exact False.elim (hx hx0)
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_1.inr.inr.inl
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        hy0 : Eq (Neg.neg y) 0
        ⊢ Exists fun g => And (Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (Matrix. …
      -/
    · exact False.elim (hy (neg_eq_zero.1 hy0))
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_1.inr.inr.inr.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        r₁ r₂ : R
        hr₁ : LT.lt 0 r₁
        left✝ : LT.lt 0 r₂
        h : Eq (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ (Neg.neg y))
        ⊢ Exists fun g => And (Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (Matrix. …
      -/
    · refine ⟨![r₁, r₂], ?_⟩
      /-
        case neg.refine_1.inr.inr.inr.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        r₁ r₂ : R
        hr₁ : LT.lt 0 r₁
        left✝ : LT.lt 0 r₂
        h : Eq (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ (Neg.neg y))
        ⊢ And (Eq (Finset.univ.sum fun i => HSMul.hSMul (Matrix.vecCons r₁ (Matrix.vec …
      -/
      rw [Fin.sum_univ_two, Fin.exists_fin_two]
      /-
        case neg.refine_1.inr.inr.inr.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        r₁ r₂ : R
        hr₁ : LT.lt 0 r₁
        left✝ : LT.lt 0 r₂
        h : Eq (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ (Neg.neg y))
        ⊢ And (Eq (HAdd.hAdd (HSMul.hSMul (Matrix.vecCons r₁ (Matrix.vecCons r₂ Matrix …
      -/
      simp [h, hr₁.ne.symm]
      /-
        🎉 no goals
      -/
    /-
      case neg.refine_2
      R : Type u_1
      inst✝³ : LinearOrderedCommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : NoZeroSMulDivisors R M
      x y : M
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      h : Exists fun g => And (Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (Matri …
      ⊢ Or (SameRay R x y) (SameRay R x (Neg.neg y))
    -/
  · rcases h with ⟨m, hm, hmne⟩
    rw [Fin.sum_univ_two, add_eq_zero_iff_eq_neg, Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.head_cons] at hm
    /-
      case neg.refine_2.intro.intro
      R : Type u_1
      inst✝³ : LinearOrderedCommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : NoZeroSMulDivisors R M
      x y : M
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      m : Fin (Nat.succ 0).succ → R
      hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
      hmne : Exists fun i => Ne (m i) 0
      ⊢ Or (SameRay R x y) (SameRay R x (Neg.neg y))
    -/
    rcases lt_trichotomy (m 0) 0 with (hm0 | hm0 | hm0) <;>
      /-
        case neg.refine_2.intro.intro.inl
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : LT.lt (m 0) 0
        ⊢ Or (SameRay R x y) (SameRay R x (Neg.neg y))
      -/
      rcases lt_trichotomy (m 1) 0 with (hm1 | hm1 | hm1)
    · refine
        Or.inr (Or.inr (Or.inr ⟨-m 0, -m 1, Left.neg_pos_iff.2 hm0, Left.neg_pos_iff.2 hm1, ?_⟩))
      /-
        case neg.refine_2.intro.intro.inl.inl
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : LT.lt (m 0) 0
        hm1 : LT.lt (m 1) 0
        ⊢ Eq (HSMul.hSMul (Neg.neg (m 0)) x) (HSMul.hSMul (Neg.neg (m 1)) (Neg.neg y))
      -/
      linear_combination (norm := module) -hm
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2.intro.intro.inl.inr.inl
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : LT.lt (m 0) 0
        hm1 : Eq (m 1) 0
        ⊢ Or (SameRay R x y) (SameRay R x (Neg.neg y))
      -/
    · exfalso
      /-
        case neg.refine_2.intro.intro.inl.inr.inl
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : LT.lt (m 0) 0
        hm1 : Eq (m 1) 0
        ⊢ False
      -/
      simp [hm1, hx, hm0.ne] at hm
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2.intro.intro.inl.inr.inr
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : LT.lt (m 0) 0
        hm1 : LT.lt 0 (m 1)
        ⊢ Or (SameRay R x y) (SameRay R x (Neg.neg y))
      -/
    · refine Or.inl (Or.inr (Or.inr ⟨-m 0, m 1, Left.neg_pos_iff.2 hm0, hm1, ?_⟩))
      /-
        case neg.refine_2.intro.intro.inl.inr.inr
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : LT.lt (m 0) 0
        hm1 : LT.lt 0 (m 1)
        ⊢ Eq (HSMul.hSMul (Neg.neg (m 0)) x) (HSMul.hSMul (m 1) y)
      -/
      linear_combination (norm := module) -hm
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2.intro.intro.inr.inl.inl
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : Eq (m 0) 0
        hm1 : LT.lt (m 1) 0
        ⊢ Or (SameRay R x y) (SameRay R x (Neg.neg y))
      -/
    · exfalso
      /-
        case neg.refine_2.intro.intro.inr.inl.inl
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : Eq (m 0) 0
        hm1 : LT.lt (m 1) 0
        ⊢ False
      -/
      simp [hm0, hy, hm1.ne] at hm
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2.intro.intro.inr.inl.inr.inl
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : Eq (m 0) 0
        hm1 : Eq (m 1) 0
        ⊢ Or (SameRay R x y) (SameRay R x (Neg.neg y))
      -/
    · rw [Fin.exists_fin_two] at hmne
      /-
        case neg.refine_2.intro.intro.inr.inl.inr.inl
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Or (Ne (m 0) 0) (Ne (m 1) 0)
        hm0 : Eq (m 0) 0
        hm1 : Eq (m 1) 0
        ⊢ Or (SameRay R x y) (SameRay R x (Neg.neg y))
      -/
      exact False.elim (not_and_or.2 hmne ⟨hm0, hm1⟩)
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2.intro.intro.inr.inl.inr.inr
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : Eq (m 0) 0
        hm1 : LT.lt 0 (m 1)
        ⊢ Or (SameRay R x y) (SameRay R x (Neg.neg y))
      -/
    · exfalso
      /-
        case neg.refine_2.intro.intro.inr.inl.inr.inr
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : Eq (m 0) 0
        hm1 : LT.lt 0 (m 1)
        ⊢ False
      -/
      simp [hm0, hy, hm1.ne.symm] at hm
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2.intro.intro.inr.inr.inl
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : LT.lt 0 (m 0)
        hm1 : LT.lt (m 1) 0
        ⊢ Or (SameRay R x y) (SameRay R x (Neg.neg y))
      -/
    · refine Or.inl (Or.inr (Or.inr ⟨m 0, -m 1, hm0, Left.neg_pos_iff.2 hm1, ?_⟩))
      /-
        case neg.refine_2.intro.intro.inr.inr.inl
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : LT.lt 0 (m 0)
        hm1 : LT.lt (m 1) 0
        ⊢ Eq (HSMul.hSMul (m 0) x) (HSMul.hSMul (Neg.neg (m 1)) y)
      -/
      rwa [neg_smul]
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2.intro.intro.inr.inr.inr.inl
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : LT.lt 0 (m 0)
        hm1 : Eq (m 1) 0
        ⊢ Or (SameRay R x y) (SameRay R x (Neg.neg y))
      -/
    · exfalso
      /-
        case neg.refine_2.intro.intro.inr.inr.inr.inl
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : LT.lt 0 (m 0)
        hm1 : Eq (m 1) 0
        ⊢ False
      -/
      simp [hm1, hx, hm0.ne.symm] at hm
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2.intro.intro.inr.inr.inr.inr
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : LT.lt 0 (m 0)
        hm1 : LT.lt 0 (m 1)
        ⊢ Or (SameRay R x y) (SameRay R x (Neg.neg y))
      -/
    · refine Or.inr (Or.inr (Or.inr ⟨m 0, m 1, hm0, hm1, ?_⟩))
      /-
        case neg.refine_2.intro.intro.inr.inr.inr.inr
        R : Type u_1
        inst✝³ : LinearOrderedCommRing R
        M : Type u_2
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        x y : M
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        m : Fin (Nat.succ 0).succ → R
        hm : Eq (HSMul.hSMul (m 0) x) (Neg.neg (HSMul.hSMul (m 1) y))
        hmne : Exists fun i => Ne (m i) 0
        hm0 : LT.lt 0 (m 0)
        hm1 : LT.lt 0 (m 1)
        ⊢ Eq (HSMul.hSMul (m 0) x) (HSMul.hSMul (m 1) (Neg.neg y))
      -/
      rwa [smul_neg]
      /-
        🎉 no goals
      -/


/-- Two vectors are in the same ray, or they are nonzero and the first is in the same ray as the
negation of the second, if and only if they are not linearly independent. -/
theorem sameRay_or_ne_zero_and_sameRay_neg_iff_not_linearIndependent {x y : M} :
    SameRay R x y ∨ x ≠ 0 ∧ y ≠ 0 ∧ SameRay R x (-y) ↔ ¬LinearIndependent R ![x, y] := by
  /-
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x y : M
    ⊢ Iff (Or (SameRay R x y) (And (Ne x 0) (And (Ne y 0) (SameRay R x (Neg.neg y) …
  -/
  rw [← sameRay_or_sameRay_neg_iff_not_linearIndependent]
  /-
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x y : M
    ⊢ Iff (Or (SameRay R x y) (And (Ne x 0) (And (Ne y 0) (SameRay R x (Neg.neg y) …
  -/
  by_cases hx : x = 0; · simp [hx]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x y : M
    hx : Not (Eq x 0)
    ⊢ Iff (Or (SameRay R x y) (And (Ne x 0) (And (Ne y 0) (SameRay R x (Neg.neg y) …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hy : y = 0 <;> simp [hx, hy]
                          /-
                            🎉 no goals
                          -/


theorem exists_pos_left (h : SameRay R x y) (hx : x ≠ 0) (hy : y ≠ 0) :
    ∃ r : R, 0 < r ∧ r • x = y :=
  let ⟨r₁, r₂, hr₁, hr₂, h⟩ := h.exists_pos hx hy
                                              /-
                                                R : Type u_1
                                                inst✝² : LinearOrderedField R
                                                M : Type u_2
                                                inst✝¹ : AddCommGroup M
                                                inst✝ : Module R M
                                                x y : M
                                                h✝ : SameRay R x y
                                                hx : Ne x 0
                                                hy : Ne y 0
                                                r₁ r₂ : R
                                                hr₁ : LT.lt 0 r₁
                                                hr₂ : LT.lt 0 r₂
                                                h : Eq (HSMul.hSMul r₁ x) (HSMul.hSMul r₂ y)
                                                ⊢ Eq (HSMul.hSMul (HMul.hMul (Inv.inv r₂) r₁) x) y
                                              -/
  ⟨r₂⁻¹ * r₁, mul_pos (inv_pos.2 hr₂) hr₁, by rw [mul_smul, h, inv_smul_smul₀ hr₂.ne']⟩
                                              /-
                                                🎉 no goals
                                              -/


theorem exists_pos_right (h : SameRay R x y) (hx : x ≠ 0) (hy : y ≠ 0) :
    ∃ r : R, 0 < r ∧ x = r • y :=
  (h.symm.exists_pos_left hy hx).imp fun _ => And.imp_right Eq.symm


/-- If a vector `v₂` is on the same ray as a nonzero vector `v₁`, then it is equal to `c • v₁` for
some nonnegative `c`. -/
theorem exists_nonneg_left (h : SameRay R x y) (hx : x ≠ 0) : ∃ r : R, 0 ≤ r ∧ r • x = y := by
  /-
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : SameRay R x y
    hx : Ne x 0
    ⊢ Exists fun r => And (LE.le 0 r) (Eq (HSMul.hSMul r x) y)
  -/
  obtain rfl | hy := eq_or_ne y 0
    /-
      case inl
      R : Type u_1
      inst✝² : LinearOrderedField R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x : M
      hx : Ne x 0
      h : SameRay R x 0
      ⊢ Exists fun r => And (LE.le 0 r) (Eq (HSMul.hSMul r x) 0)
    -/
  · exact ⟨0, le_rfl, zero_smul _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝² : LinearOrderedField R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x y : M
      h : SameRay R x y
      hx : Ne x 0
      hy : Ne y 0
      ⊢ Exists fun r => And (LE.le 0 r) (Eq (HSMul.hSMul r x) y)
    -/
  · exact (h.exists_pos_left hx hy).imp fun _ => And.imp_left le_of_lt
    /-
      🎉 no goals
    -/


/-- If a vector `v₁` is on the same ray as a nonzero vector `v₂`, then it is equal to `c • v₂` for
some nonnegative `c`. -/
theorem exists_nonneg_right (h : SameRay R x y) (hy : y ≠ 0) : ∃ r : R, 0 ≤ r ∧ x = r • y :=
  (h.symm.exists_nonneg_left hy).imp fun _ => And.imp_right Eq.symm


/-- If vectors `v₁` and `v₂` are on the same ray, then for some nonnegative `a b`, `a + b = 1`, we
have `v₁ = a • (v₁ + v₂)` and `v₂ = b • (v₁ + v₂)`. -/
theorem exists_eq_smul_add (h : SameRay R v₁ v₂) :
    ∃ a b : R, 0 ≤ a ∧ 0 ≤ b ∧ a + b = 1 ∧ v₁ = a • (v₁ + v₂) ∧ v₂ = b • (v₁ + v₂) := by
  /-
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    v₁ v₂ : M
    h : SameRay R v₁ v₂
    ⊢ Exists fun a => Exists fun b => And (LE.le 0 a) (And (LE.le 0 b) (And (Eq (H …
  -/
  rcases h with (rfl | rfl | ⟨r₁, r₂, h₁, h₂, H⟩)
    /-
      case inl
      R : Type u_1
      inst✝² : LinearOrderedField R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v₂ : M
      ⊢ Exists fun a => Exists fun b => And (LE.le 0 a) (And (LE.le 0 b) (And (Eq (H …
    -/
  · use 0, 1
    /-
      case h
      R : Type u_1
      inst✝² : LinearOrderedField R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v₂ : M
      ⊢ And (LE.le 0 0) (And (LE.le 0 1) (And (Eq (HAdd.hAdd 0 1) 1) (And (Eq 0 (HSM …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      R : Type u_1
      inst✝² : LinearOrderedField R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v₁ : M
      ⊢ Exists fun a => Exists fun b => And (LE.le 0 a) (And (LE.le 0 b) (And (Eq (H …
    -/
  · use 1, 0
    /-
      case h
      R : Type u_1
      inst✝² : LinearOrderedField R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v₁ : M
      ⊢ And (LE.le 0 1) (And (LE.le 0 0) (And (Eq (HAdd.hAdd 1 0) 1) (And (Eq v₁ (HS …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.intro.intro.intro.intro
      R : Type u_1
      inst✝² : LinearOrderedField R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v₁ v₂ : M
      r₁ r₂ : R
      h₁ : LT.lt 0 r₁
      h₂ : LT.lt 0 r₂
      H : Eq (HSMul.hSMul r₁ v₁) (HSMul.hSMul r₂ v₂)
      ⊢ Exists fun a => Exists fun b => And (LE.le 0 a) (And (LE.le 0 b) (And (Eq (H …
    -/
  · have h₁₂ : 0 < r₁ + r₂ := add_pos h₁ h₂
    refine
      ⟨r₂ / (r₁ + r₂), r₁ / (r₁ + r₂), div_nonneg h₂.le h₁₂.le, div_nonneg h₁.le h₁₂.le, ?_, ?_, ?_⟩
      /-
        case inr.inr.intro.intro.intro.intro.refine_1
        R : Type u_1
        inst✝² : LinearOrderedField R
        M : Type u_2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        v₁ v₂ : M
        r₁ r₂ : R
        h₁ : LT.lt 0 r₁
        h₂ : LT.lt 0 r₂
        H : Eq (HSMul.hSMul r₁ v₁) (HSMul.hSMul r₂ v₂)
        h₁₂ : LT.lt 0 (HAdd.hAdd r₁ r₂)
        ⊢ Eq (HAdd.hAdd (HDiv.hDiv r₂ (HAdd.hAdd r₁ r₂)) (HDiv.hDiv r₁ (HAdd.hAdd r₁ r …
      -/
    · rw [← add_div, add_comm, div_self h₁₂.ne']
      /-
        🎉 no goals
      -/
      /-
        case inr.inr.intro.intro.intro.intro.refine_2
        R : Type u_1
        inst✝² : LinearOrderedField R
        M : Type u_2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        v₁ v₂ : M
        r₁ r₂ : R
        h₁ : LT.lt 0 r₁
        h₂ : LT.lt 0 r₂
        H : Eq (HSMul.hSMul r₁ v₁) (HSMul.hSMul r₂ v₂)
        h₁₂ : LT.lt 0 (HAdd.hAdd r₁ r₂)
        ⊢ Eq v₁ (HSMul.hSMul (HDiv.hDiv r₂ (HAdd.hAdd r₁ r₂)) (HAdd.hAdd v₁ v₂))
      -/
    · rw [div_eq_inv_mul, mul_smul, smul_add, ← H, ← add_smul, add_comm r₂, inv_smul_smul₀ h₁₂.ne']
      /-
        🎉 no goals
      -/
      /-
        case inr.inr.intro.intro.intro.intro.refine_3
        R : Type u_1
        inst✝² : LinearOrderedField R
        M : Type u_2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        v₁ v₂ : M
        r₁ r₂ : R
        h₁ : LT.lt 0 r₁
        h₂ : LT.lt 0 r₂
        H : Eq (HSMul.hSMul r₁ v₁) (HSMul.hSMul r₂ v₂)
        h₁₂ : LT.lt 0 (HAdd.hAdd r₁ r₂)
        ⊢ Eq v₂ (HSMul.hSMul (HDiv.hDiv r₁ (HAdd.hAdd r₁ r₂)) (HAdd.hAdd v₁ v₂))
      -/
    · rw [div_eq_inv_mul, mul_smul, smul_add, H, ← add_smul, add_comm r₂, inv_smul_smul₀ h₁₂.ne']
      /-
        🎉 no goals
      -/


/-- If vectors `v₁` and `v₂` are on the same ray, then they are nonnegative multiples of the same
vector. Actually, this vector can be assumed to be `v₁ + v₂`, see `SameRay.exists_eq_smul_add`. -/
theorem exists_eq_smul (h : SameRay R v₁ v₂) :
    ∃ (u : M) (a b : R), 0 ≤ a ∧ 0 ≤ b ∧ a + b = 1 ∧ v₁ = a • u ∧ v₂ = b • u :=
  ⟨v₁ + v₂, h.exists_eq_smul_add⟩


theorem exists_pos_left_iff_sameRay (hx : x ≠ 0) (hy : y ≠ 0) :
    (∃ r : R, 0 < r ∧ r • x = y) ↔ SameRay R x y := by
  /-
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (Exists fun r => And (LT.lt 0 r) (Eq (HSMul.hSMul r x) y)) (SameRay R x y)
  -/
  refine ⟨fun h => ?_, fun h => h.exists_pos_left hx hy⟩
  /-
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    hx : Ne x 0
    hy : Ne y 0
    h : Exists fun r => And (LT.lt 0 r) (Eq (HSMul.hSMul r x) y)
    ⊢ SameRay R x y
  -/
  rcases h with ⟨r, hr, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x : M
    hx : Ne x 0
    r : R
    hr : LT.lt 0 r
    hy : Ne (HSMul.hSMul r x) 0
    ⊢ SameRay R x (HSMul.hSMul r x)
  -/
  exact SameRay.sameRay_pos_smul_right x hr
  /-
    🎉 no goals
  -/


theorem exists_pos_left_iff_sameRay_and_ne_zero (hx : x ≠ 0) :
    (∃ r : R, 0 < r ∧ r • x = y) ↔ SameRay R x y ∧ y ≠ 0 := by
  /-
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    hx : Ne x 0
    ⊢ Iff (Exists fun r => And (LT.lt 0 r) (Eq (HSMul.hSMul r x) y)) (And (SameRay …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝² : LinearOrderedField R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x y : M
      hx : Ne x 0
      ⊢ (Exists fun r => And (LT.lt 0 r) (Eq (HSMul.hSMul r x) y)) → And (SameRay R  …
    -/
  · rintro ⟨r, hr, rfl⟩
    /-
      case mp.intro.intro
      R : Type u_1
      inst✝² : LinearOrderedField R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x : M
      hx : Ne x 0
      r : R
      hr : LT.lt 0 r
      ⊢ And (SameRay R x (HSMul.hSMul r x)) (Ne (HSMul.hSMul r x) 0)
    -/
    simp [hx, hr.le, hr.ne']
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝² : LinearOrderedField R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x y : M
      hx : Ne x 0
      ⊢ And (SameRay R x y) (Ne y 0) → Exists fun r => And (LT.lt 0 r) (Eq (HSMul.hS …
    -/
  · rintro ⟨hxy, hy⟩
    /-
      case mpr.intro
      R : Type u_1
      inst✝² : LinearOrderedField R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x y : M
      hx : Ne x 0
      hxy : SameRay R x y
      hy : Ne y 0
      ⊢ Exists fun r => And (LT.lt 0 r) (Eq (HSMul.hSMul r x) y)
    -/
    exact (exists_pos_left_iff_sameRay hx hy).2 hxy
    /-
      🎉 no goals
    -/


theorem exists_nonneg_left_iff_sameRay (hx : x ≠ 0) :
    (∃ r : R, 0 ≤ r ∧ r • x = y) ↔ SameRay R x y := by
  /-
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    hx : Ne x 0
    ⊢ Iff (Exists fun r => And (LE.le 0 r) (Eq (HSMul.hSMul r x) y)) (SameRay R x y)
  -/
  refine ⟨fun h => ?_, fun h => h.exists_nonneg_left hx⟩
  /-
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    hx : Ne x 0
    h : Exists fun r => And (LE.le 0 r) (Eq (HSMul.hSMul r x) y)
    ⊢ SameRay R x y
  -/
  rcases h with ⟨r, hr, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x : M
    hx : Ne x 0
    r : R
    hr : LE.le 0 r
    ⊢ SameRay R x (HSMul.hSMul r x)
  -/
  exact SameRay.sameRay_nonneg_smul_right x hr
  /-
    🎉 no goals
  -/


theorem exists_pos_right_iff_sameRay (hx : x ≠ 0) (hy : y ≠ 0) :
    (∃ r : R, 0 < r ∧ x = r • y) ↔ SameRay R x y := by
  /-
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (Exists fun r => And (LT.lt 0 r) (Eq x (HSMul.hSMul r y))) (SameRay R x y)
  -/
  rw [SameRay.sameRay_comm]
  /-
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (Exists fun r => And (LT.lt 0 r) (Eq x (HSMul.hSMul r y))) (SameRay R y x)
  -/
  simp_rw [eq_comm (a := x)]
  /-
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (Exists fun r => And (LT.lt 0 r) (Eq (HSMul.hSMul r y) x)) (SameRay R y x)
  -/
  exact exists_pos_left_iff_sameRay hy hx
  /-
    🎉 no goals
  -/


theorem exists_pos_right_iff_sameRay_and_ne_zero (hy : y ≠ 0) :
    (∃ r : R, 0 < r ∧ x = r • y) ↔ SameRay R x y ∧ x ≠ 0 := by
  /-
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    hy : Ne y 0
    ⊢ Iff (Exists fun r => And (LT.lt 0 r) (Eq x (HSMul.hSMul r y))) (And (SameRay …
  -/
  rw [SameRay.sameRay_comm]
  /-
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    hy : Ne y 0
    ⊢ Iff (Exists fun r => And (LT.lt 0 r) (Eq x (HSMul.hSMul r y))) (And (SameRay …
  -/
  simp_rw [eq_comm (a := x)]
  /-
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    hy : Ne y 0
    ⊢ Iff (Exists fun r => And (LT.lt 0 r) (Eq (HSMul.hSMul r y) x)) (And (SameRay …
  -/
  exact exists_pos_left_iff_sameRay_and_ne_zero hy
  /-
    🎉 no goals
  -/


theorem exists_nonneg_right_iff_sameRay (hy : y ≠ 0) :
    (∃ r : R, 0 ≤ r ∧ x = r • y) ↔ SameRay R x y := by
  /-
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    hy : Ne y 0
    ⊢ Iff (Exists fun r => And (LE.le 0 r) (Eq x (HSMul.hSMul r y))) (SameRay R x y)
  -/
  rw [SameRay.sameRay_comm]
  /-
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    hy : Ne y 0
    ⊢ Iff (Exists fun r => And (LE.le 0 r) (Eq x (HSMul.hSMul r y))) (SameRay R y x)
  -/
  simp_rw [eq_comm (a := x)]
  /-
    R : Type u_1
    inst✝² : LinearOrderedField R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    hy : Ne y 0
    ⊢ Iff (Exists fun r => And (LE.le 0 r) (Eq (HSMul.hSMul r y) x)) (SameRay R y x)
  -/
  exact exists_nonneg_left_iff_sameRay (R := R) hy
  /-
    🎉 no goals
  -/


