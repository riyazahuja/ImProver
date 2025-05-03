/-- A vector `x` is reflective with respect to a bilinear form if multiplication by its norm is
injective, and for any vector `y`, the norm of `x` divides twice the inner product of `x` and `y`.
These conditions are what we need when describing reflection as a map taking `y` to
`y - 2 • (B x y) / (B x x) • x`. -/
structure IsReflective (B : M →ₗ[R] M →ₗ[R] R) (x : M) : Prop where
  regular : IsRegular (B x x)
  dvd_two_mul : ∀ y, B x x ∣ 2 * B x y


lemma of_dvd_two [IsDomain R] [NeZero (2 : R)] (hx : B x x ∣ 2) :
    IsReflective B x where
                                                     /-
                                                       R : Type u_1
                                                       M : Type u_2
                                                       inst✝⁴ : CommRing R
                                                       inst✝³ : AddCommGroup M
                                                       inst✝² : Module R M
                                                       B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
                                                       x : M
                                                       inst✝¹ : IsDomain R
                                                       inst✝ : NeZero 2
                                                       hx : Dvd.dvd ((B x) x) 2
                                                       contra : Eq ((B x) x) 0
                                                       ⊢ False
                                                     -/
  regular := isRegular_of_ne_zero <| fun contra ↦ by simp [contra, two_ne_zero (α := R)] at hx
                                                     /-
                                                       🎉 no goals
                                                     -/
  dvd_two_mul y := hx.mul_right (B x y)


/-- The coroot attached to a reflective vector. -/
def coroot : M →ₗ[R] R where
  toFun y := (hx.2 y).choose
  map_add' a b := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      x : M
      hx : B.IsReflective x
      a b : M
      ⊢ Eq ((fun y => Exists.choose ⋯) (HAdd.hAdd a b)) (HAdd.hAdd ((fun y => Exists …
    -/
    refine hx.1.1 ?_
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      x : M
      hx : B.IsReflective x
      a b : M
      ⊢ Eq ((fun x_1 => HMul.hMul ((B x) x) x_1) ((fun y => Exists.choose ⋯) (HAdd.h …
    -/
    simp only
    rw [← (hx.2 (a + b)).choose_spec, mul_add, ← (hx.2 a).choose_spec, ← (hx.2 b).choose_spec,
      map_add, mul_add]
  map_smul' r a := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      x : M
      hx : B.IsReflective x
      r : R
      a : M
      ⊢ Eq ({ toFun := fun y => Exists.choose ⋯, map_add' := ⋯ }.toFun (HSMul.hSMul  …
    -/
    refine hx.1.1 ?_
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      x : M
      hx : B.IsReflective x
      r : R
      a : M
      ⊢ Eq ((fun x_1 => HMul.hMul ((B x) x) x_1) ({ toFun := fun y => Exists.choose  …
    -/
    simp only [RingHom.id_apply]
    rw [← (hx.2 (r • a)).choose_spec, smul_eq_mul, mul_left_comm, ← (hx.2 a).choose_spec, map_smul,
      two_mul, smul_eq_mul, two_mul, mul_add]


@[simp]
lemma apply_self_mul_coroot_apply {y : M} : B x x * coroot B hx y = 2 * B x y :=
  (hx.dvd_two_mul y).choose_spec.symm


@[simp]
lemma smul_coroot : B x x • coroot B hx = 2 • B x := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    x : M
    hx : B.IsReflective x
    ⊢ Eq (HSMul.hSMul ((B x) x) (LinearMap.IsReflective.coroot B hx)) (HSMul.hSMul …
  -/
  ext y
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    x : M
    hx : B.IsReflective x
    y : M
    ⊢ Eq ((HSMul.hSMul ((B x) x) (LinearMap.IsReflective.coroot B hx)) y) ((HSMul. …
  -/
  simp [smul_apply, smul_eq_mul, nsmul_eq_mul, Nat.cast_ofNat, apply_self_mul_coroot_apply]
  /-
    🎉 no goals
  -/


@[simp]
lemma coroot_apply_self : coroot B hx x = 2 :=
                        /-
                          R : Type u_1
                          M : Type u_2
                          inst✝² : CommRing R
                          inst✝¹ : AddCommGroup M
                          inst✝ : Module R M
                          B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
                          x : M
                          hx : B.IsReflective x
                          ⊢ Eq ((fun x_1 => HMul.hMul ((B x) x) x_1) ((LinearMap.IsReflective.coroot B h …
                        -/
  hx.regular.left <| by simp [mul_comm _ (B x x)]
                        /-
                          🎉 no goals
                        -/


lemma isOrthogonal_reflection (hSB : LinearMap.IsSymm B) :
    B.IsOrthogonal (Module.reflection (coroot_apply_self B hx)) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    x : M
    hx : B.IsReflective x
    hSB : B.IsSymm
    ⊢ B.IsOrthogonal ⇑(Module.reflection ⋯)
  -/
  intro y z
  simp only [LinearEquiv.coe_coe, reflection_apply, LinearMap.map_sub, map_smul, sub_apply,
    smul_apply, smul_eq_mul]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    x : M
    hx : B.IsReflective x
    hSB : B.IsSymm
    y z : M
    ⊢ Eq (HSub.hSub (HSub.hSub ((B y) z) (HMul.hMul ((LinearMap.IsReflective.coroo …
  -/
  refine hx.1.1 ?_
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    x : M
    hx : B.IsReflective x
    hSB : B.IsSymm
    y z : M
    ⊢ Eq ((fun x_1 => HMul.hMul ((B x) x) x_1) (HSub.hSub (HSub.hSub ((B y) z) (HM …
  -/
  simp only [mul_sub, ← mul_assoc, apply_self_mul_coroot_apply]
  rw [sub_eq_iff_eq_add, ← hSB x y, RingHom.id_apply, mul_assoc _ _ (B x x), mul_comm _ (B x x),
    apply_self_mul_coroot_apply]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    x : M
    hx : B.IsReflective x
    hSB : B.IsSymm
    y z : M
    ⊢ Eq (HSub.hSub (HMul.hMul ((B x) x) ((B y) z)) (HMul.hMul (HMul.hMul 2 ((B x) …
  -/
  ring
  /-
    🎉 no goals
  -/


lemma reflective_reflection (hSB : LinearMap.IsSymm B) {y : M}
    (hx : IsReflective B x) (hy : IsReflective B y) :
    IsReflective B (Module.reflection (coroot_apply_self B hx) y) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    x : M
    hSB : B.IsSymm
    y : M
    hx : B.IsReflective x
    hy : B.IsReflective y
    ⊢ B.IsReflective ((Module.reflection ⋯) y)
  -/
  constructor
    /-
      case regular
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      x : M
      hSB : B.IsSymm
      y : M
      hx : B.IsReflective x
      hy : B.IsReflective y
      ⊢ IsRegular ((B ((Module.reflection ⋯) y)) ((Module.reflection ⋯) y))
    -/
  · rw [isOrthogonal_reflection B hx hSB]
    /-
      case regular
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      x : M
      hSB : B.IsSymm
      y : M
      hx : B.IsReflective x
      hy : B.IsReflective y
      ⊢ IsRegular ((B y) y)
    -/
    exact hy.1
    /-
      🎉 no goals
    -/
    /-
      case dvd_two_mul
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      x : M
      hSB : B.IsSymm
      y : M
      hx : B.IsReflective x
      hy : B.IsReflective y
      ⊢ ∀ (y_1 : M), Dvd.dvd ((B ((Module.reflection ⋯) y)) ((Module.reflection ⋯) y …
    -/
  · intro z
    have hz : Module.reflection (coroot_apply_self B hx)
        (Module.reflection (coroot_apply_self B hx) z) = z := by
      exact (LinearEquiv.eq_symm_apply (Module.reflection (coroot_apply_self B hx))).mp rfl
    rw [← hz, isOrthogonal_reflection B hx hSB,
      isOrthogonal_reflection B hx hSB]
    /-
      case dvd_two_mul
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      x : M
      hSB : B.IsSymm
      y : M
      hx : B.IsReflective x
      hy : B.IsReflective y
      z : M
      hz : Eq ((Module.reflection ⋯) ((Module.reflection ⋯) z)) z
      ⊢ Dvd.dvd ((B y) y) (HMul.hMul 2 ((B y) ((Module.reflection ⋯) z)))
    -/
    exact hy.2 _
    /-
      🎉 no goals
    -/


/-- The root pairing given by all reflective vectors for a bilinear form. -/
def ofBilinear [IsReflexive R M] (B : M →ₗ[R] M →ₗ[R] R) (hNB : LinearMap.Nondegenerate B)
    (hSB : LinearMap.IsSymm B) (h2 : IsRegular (2 : R)) :
    RootPairing {x : M | IsReflective B x} R M (Dual R M) where
  toPerfectPairing := (IsReflexive.toPerfectPairingDual (R := R) (M := M)).flip
  root := Embedding.subtype fun x ↦ IsReflective B x
  coroot :=
    { toFun := fun x => IsReflective.coroot B x.2
      inj' := by
        /-
          R : Type u_1
          M : Type u_2
          inst✝³ : CommRing R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module.IsReflexive R M
          B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
          hNB : B.Nondegenerate
          hSB : B.IsSymm
          h2 : IsRegular 2
          ⊢ Function.Injective fun x => LinearMap.IsReflective.coroot B ⋯
        -/
        intro x y hxy
        /-
          R : Type u_1
          M : Type u_2
          inst✝³ : CommRing R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module.IsReflexive R M
          B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
          hNB : B.Nondegenerate
          hSB : B.IsSymm
          h2 : IsRegular 2
          x y : ↑(setOf fun x => B.IsReflective x)
          hxy : Eq ((fun x => LinearMap.IsReflective.coroot B ⋯) x) ((fun x => LinearMap …
          ⊢ Eq x y
        -/
        simp only [mem_setOf_eq] at hxy -- x* = y*
        have h1 : ∀ z, IsReflective.coroot B x.2 z = IsReflective.coroot B y.2 z :=
          fun z => congrFun (congrArg DFunLike.coe hxy) z
        have h2x : ∀ z, B x x * IsReflective.coroot B x.2 z =
            B x x * IsReflective.coroot B y.2 z :=
          fun z => congrArg (HMul.hMul ((B x) x)) (h1 z)
        have h2y : ∀ z, B y y * IsReflective.coroot B x.2 z =
            B y y * IsReflective.coroot B y.2 z :=
          fun z => congrArg (HMul.hMul ((B y) y)) (h1 z)
        /-
          R : Type u_1
          M : Type u_2
          inst✝³ : CommRing R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module.IsReflexive R M
          B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
          hNB : B.Nondegenerate
          hSB : B.IsSymm
          h2 : IsRegular 2
          x y : ↑(setOf fun x => B.IsReflective x)
          hxy : Eq (LinearMap.IsReflective.coroot B ⋯) (LinearMap.IsReflective.coroot B ⋯)
          h1 : ∀ (z : M), Eq ((LinearMap.IsReflective.coroot B ⋯) z) ((LinearMap.IsRefle …
          h2x : ∀ (z : M), Eq (HMul.hMul ((B ↑x) ↑x) ((LinearMap.IsReflective.coroot B ⋯ …
          h2y : ∀ (z : M), Eq (HMul.hMul ((B ↑y) ↑y) ((LinearMap.IsReflective.coroot B ⋯ …
          ⊢ Eq x y
        -/
        simp_rw [apply_self_mul_coroot_apply B x.2] at h2x -- 2(x,z) = (x,x)y*(z)
        /-
          R : Type u_1
          M : Type u_2
          inst✝³ : CommRing R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module.IsReflexive R M
          B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
          hNB : B.Nondegenerate
          hSB : B.IsSymm
          h2 : IsRegular 2
          x y : ↑(setOf fun x => B.IsReflective x)
          hxy : Eq (LinearMap.IsReflective.coroot B ⋯) (LinearMap.IsReflective.coroot B ⋯)
          h1 : ∀ (z : M), Eq ((LinearMap.IsReflective.coroot B ⋯) z) ((LinearMap.IsRefle …
          h2y : ∀ (z : M), Eq (HMul.hMul ((B ↑y) ↑y) ((LinearMap.IsReflective.coroot B ⋯ …
          h2x : ∀ (z : M), Eq (HMul.hMul 2 ((B ↑x) z)) (HMul.hMul ((B ↑x) ↑x) ((LinearMa …
          ⊢ Eq x y
        -/
        simp_rw [apply_self_mul_coroot_apply B y.2] at h2y -- (y,y)x*(z) = 2(y,z)
        have h2xy : B x x = B y y := by
          refine h2.1 ?_
          dsimp only
          specialize h2x y
          rw [coroot_apply_self] at h2x
          specialize h2y x
          rw [coroot_apply_self] at h2y
          rw [mul_comm, ← h2x, ← hSB, RingHom.id_apply, ← h2y, mul_comm]
        /-
          R : Type u_1
          M : Type u_2
          inst✝³ : CommRing R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module.IsReflexive R M
          B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
          hNB : B.Nondegenerate
          hSB : B.IsSymm
          h2 : IsRegular 2
          x y : ↑(setOf fun x => B.IsReflective x)
          hxy : Eq (LinearMap.IsReflective.coroot B ⋯) (LinearMap.IsReflective.coroot B ⋯)
          h1 : ∀ (z : M), Eq ((LinearMap.IsReflective.coroot B ⋯) z) ((LinearMap.IsRefle …
          h2x : ∀ (z : M), Eq (HMul.hMul 2 ((B ↑x) z)) (HMul.hMul ((B ↑x) ↑x) ((LinearMa …
          h2y : ∀ (z : M), Eq (HMul.hMul ((B ↑y) ↑y) ((LinearMap.IsReflective.coroot B ⋯ …
          h2xy : Eq ((B ↑x) ↑x) ((B ↑y) ↑y)
          ⊢ Eq x y
        -/
        rw [Subtype.ext_iff_val, ← sub_eq_zero]
        /-
          R : Type u_1
          M : Type u_2
          inst✝³ : CommRing R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module.IsReflexive R M
          B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
          hNB : B.Nondegenerate
          hSB : B.IsSymm
          h2 : IsRegular 2
          x y : ↑(setOf fun x => B.IsReflective x)
          hxy : Eq (LinearMap.IsReflective.coroot B ⋯) (LinearMap.IsReflective.coroot B ⋯)
          h1 : ∀ (z : M), Eq ((LinearMap.IsReflective.coroot B ⋯) z) ((LinearMap.IsRefle …
          h2x : ∀ (z : M), Eq (HMul.hMul 2 ((B ↑x) z)) (HMul.hMul ((B ↑x) ↑x) ((LinearMa …
          h2y : ∀ (z : M), Eq (HMul.hMul ((B ↑y) ↑y) ((LinearMap.IsReflective.coroot B ⋯ …
          h2xy : Eq ((B ↑x) ↑x) ((B ↑y) ↑y)
          ⊢ Eq (HSub.hSub ↑x ↑y) 0
        -/
        refine hNB.1 _ (fun z => ?_)
        /-
          R : Type u_1
          M : Type u_2
          inst✝³ : CommRing R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module.IsReflexive R M
          B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
          hNB : B.Nondegenerate
          hSB : B.IsSymm
          h2 : IsRegular 2
          x y : ↑(setOf fun x => B.IsReflective x)
          hxy : Eq (LinearMap.IsReflective.coroot B ⋯) (LinearMap.IsReflective.coroot B ⋯)
          h1 : ∀ (z : M), Eq ((LinearMap.IsReflective.coroot B ⋯) z) ((LinearMap.IsRefle …
          h2x : ∀ (z : M), Eq (HMul.hMul 2 ((B ↑x) z)) (HMul.hMul ((B ↑x) ↑x) ((LinearMa …
          h2y : ∀ (z : M), Eq (HMul.hMul ((B ↑y) ↑y) ((LinearMap.IsReflective.coroot B ⋯ …
          h2xy : Eq ((B ↑x) ↑x) ((B ↑y) ↑y)
          z : M
          ⊢ Eq ((B (HSub.hSub ↑x ↑y)) z) 0
        -/
        rw [map_sub, LinearMap.sub_apply, sub_eq_zero]
        /-
          R : Type u_1
          M : Type u_2
          inst✝³ : CommRing R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module.IsReflexive R M
          B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
          hNB : B.Nondegenerate
          hSB : B.IsSymm
          h2 : IsRegular 2
          x y : ↑(setOf fun x => B.IsReflective x)
          hxy : Eq (LinearMap.IsReflective.coroot B ⋯) (LinearMap.IsReflective.coroot B ⋯)
          h1 : ∀ (z : M), Eq ((LinearMap.IsReflective.coroot B ⋯) z) ((LinearMap.IsRefle …
          h2x : ∀ (z : M), Eq (HMul.hMul 2 ((B ↑x) z)) (HMul.hMul ((B ↑x) ↑x) ((LinearMa …
          h2y : ∀ (z : M), Eq (HMul.hMul ((B ↑y) ↑y) ((LinearMap.IsReflective.coroot B ⋯ …
          h2xy : Eq ((B ↑x) ↑x) ((B ↑y) ↑y)
          z : M
          ⊢ Eq ((B ↑x) z) ((B ↑y) z)
        -/
        refine h2.1 ?_
        /-
          R : Type u_1
          M : Type u_2
          inst✝³ : CommRing R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module.IsReflexive R M
          B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
          hNB : B.Nondegenerate
          hSB : B.IsSymm
          h2 : IsRegular 2
          x y : ↑(setOf fun x => B.IsReflective x)
          hxy : Eq (LinearMap.IsReflective.coroot B ⋯) (LinearMap.IsReflective.coroot B ⋯)
          h1 : ∀ (z : M), Eq ((LinearMap.IsReflective.coroot B ⋯) z) ((LinearMap.IsRefle …
          h2x : ∀ (z : M), Eq (HMul.hMul 2 ((B ↑x) z)) (HMul.hMul ((B ↑x) ↑x) ((LinearMa …
          h2y : ∀ (z : M), Eq (HMul.hMul ((B ↑y) ↑y) ((LinearMap.IsReflective.coroot B ⋯ …
          h2xy : Eq ((B ↑x) ↑x) ((B ↑y) ↑y)
          z : M
          ⊢ Eq ((fun x => HMul.hMul 2 x) ((B ↑x) z)) ((fun x => HMul.hMul 2 x) ((B ↑y) z))
        -/
        dsimp only
        /-
          R : Type u_1
          M : Type u_2
          inst✝³ : CommRing R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module.IsReflexive R M
          B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
          hNB : B.Nondegenerate
          hSB : B.IsSymm
          h2 : IsRegular 2
          x y : ↑(setOf fun x => B.IsReflective x)
          hxy : Eq (LinearMap.IsReflective.coroot B ⋯) (LinearMap.IsReflective.coroot B ⋯)
          h1 : ∀ (z : M), Eq ((LinearMap.IsReflective.coroot B ⋯) z) ((LinearMap.IsRefle …
          h2x : ∀ (z : M), Eq (HMul.hMul 2 ((B ↑x) z)) (HMul.hMul ((B ↑x) ↑x) ((LinearMa …
          h2y : ∀ (z : M), Eq (HMul.hMul ((B ↑y) ↑y) ((LinearMap.IsReflective.coroot B ⋯ …
          h2xy : Eq ((B ↑x) ↑x) ((B ↑y) ↑y)
          z : M
          ⊢ Eq (HMul.hMul 2 ((B ↑x) z)) (HMul.hMul 2 ((B ↑y) z))
        -/
        rw [h2x z, ← h2y z, hxy, h2xy] }
        /-
          🎉 no goals
        -/
  root_coroot_two x := by
    dsimp only [coe_setOf, Embedding.coe_subtype, PerfectPairing.toLin_apply, mem_setOf_eq, id_eq,
      eq_mp_eq_cast, RingHom.id_apply, eq_mpr_eq_cast, cast_eq, LinearMap.sub_apply,
      Embedding.coeFn_mk, PerfectPairing.flip_apply_apply]
    /-
      R : Type u_1
      M : Type u_2
      inst✝³ : CommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Module.IsReflexive R M
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      hNB : B.Nondegenerate
      hSB : B.IsSymm
      h2 : IsRegular 2
      x : ↑(setOf fun x => B.IsReflective x)
      ⊢ Eq ((IsReflexive.toPerfectPairingDual (LinearMap.IsReflective.coroot B ⋯)) ↑ …
    -/
    exact coroot_apply_self B x.2
    /-
      🎉 no goals
    -/
  reflection_perm x :=
    { toFun := fun y => ⟨(Module.reflection (coroot_apply_self B x.2) y),
        reflective_reflection B hSB x.2 y.2⟩
      invFun := fun y => ⟨(Module.reflection (coroot_apply_self B x.2) y),
        reflective_reflection B hSB x.2 y.2⟩
      left_inv := by
        /-
          R : Type u_1
          M : Type u_2
          inst✝³ : CommRing R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module.IsReflexive R M
          B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
          hNB : B.Nondegenerate
          hSB : B.IsSymm
          h2 : IsRegular 2
          x : ↑(setOf fun x => B.IsReflective x)
          ⊢ Function.LeftInverse (fun y => ⟨(Module.reflection ⋯) ↑y, ⋯⟩) fun y => ⟨(Mod …
        -/
        intro y
        /-
          R : Type u_1
          M : Type u_2
          inst✝³ : CommRing R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module.IsReflexive R M
          B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
          hNB : B.Nondegenerate
          hSB : B.IsSymm
          h2 : IsRegular 2
          x y : ↑(setOf fun x => B.IsReflective x)
          ⊢ Eq ((fun y => ⟨(Module.reflection ⋯) ↑y, ⋯⟩) ((fun y => ⟨(Module.reflection  …
        -/
        simp [involutive_reflection (coroot_apply_self B x.2) y]
        /-
          🎉 no goals
        -/
      right_inv := by
        /-
          R : Type u_1
          M : Type u_2
          inst✝³ : CommRing R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module.IsReflexive R M
          B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
          hNB : B.Nondegenerate
          hSB : B.IsSymm
          h2 : IsRegular 2
          x : ↑(setOf fun x => B.IsReflective x)
          ⊢ Function.RightInverse (fun y => ⟨(Module.reflection ⋯) ↑y, ⋯⟩) fun y => ⟨(Mo …
        -/
        intro y
        /-
          R : Type u_1
          M : Type u_2
          inst✝³ : CommRing R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module.IsReflexive R M
          B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
          hNB : B.Nondegenerate
          hSB : B.IsSymm
          h2 : IsRegular 2
          x y : ↑(setOf fun x => B.IsReflective x)
          ⊢ Eq ((fun y => ⟨(Module.reflection ⋯) ↑y, ⋯⟩) ((fun y => ⟨(Module.reflection  …
        -/
        simp [involutive_reflection (coroot_apply_self B x.2) y] }
        /-
          🎉 no goals
        -/
  reflection_perm_root x y := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝³ : CommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Module.IsReflexive R M
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      hNB : B.Nondegenerate
      hSB : B.IsSymm
      h2 : IsRegular 2
      x y : ↑(setOf fun x => B.IsReflective x)
      ⊢ Eq (HSub.hSub ((Function.Embedding.subtype fun x => B.IsReflective x) y) (HS …
    -/
    simp [Module.reflection_apply]
    /-
      🎉 no goals
    -/
  reflection_perm_coroot x y := by
    simp only [coe_setOf, mem_setOf_eq, Embedding.coeFn_mk, Embedding.coe_subtype,
      PerfectPairing.flip_apply_apply, IsReflexive.toPerfectPairingDual_apply, Equiv.coe_fn_mk]
    /-
      R : Type u_1
      M : Type u_2
      inst✝³ : CommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Module.IsReflexive R M
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      hNB : B.Nondegenerate
      hSB : B.IsSymm
      h2 : IsRegular 2
      x y : ↑(setOf fun x => B.IsReflective x)
      ⊢ Eq (HSub.hSub (LinearMap.IsReflective.coroot B ⋯) (HSMul.hSMul ((LinearMap.I …
    -/
    ext z
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝³ : CommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Module.IsReflexive R M
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      hNB : B.Nondegenerate
      hSB : B.IsSymm
      h2 : IsRegular 2
      x y : ↑(setOf fun x => B.IsReflective x)
      z : M
      ⊢ Eq ((HSub.hSub (LinearMap.IsReflective.coroot B ⋯) (HSMul.hSMul ((LinearMap. …
    -/
    simp only [LinearMap.sub_apply, LinearMap.smul_apply, smul_eq_mul]
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝³ : CommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Module.IsReflexive R M
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      hNB : B.Nondegenerate
      hSB : B.IsSymm
      h2 : IsRegular 2
      x y : ↑(setOf fun x => B.IsReflective x)
      z : M
      ⊢ Eq (HSub.hSub ((LinearMap.IsReflective.coroot B ⋯) z) (HMul.hMul ((LinearMap …
    -/
    refine y.2.1.1 ?_
    simp only [mem_setOf_eq, PerfectPairing.flip_apply_apply, mul_sub,
      apply_self_mul_coroot_apply B y.2, ← mul_assoc]
    rw [← isOrthogonal_reflection B x.2 hSB y y, apply_self_mul_coroot_apply, ← hSB z, ← hSB z,
      RingHom.id_apply, RingHom.id_apply, Module.reflection_apply, map_sub,
      mul_sub, sub_eq_sub_iff_comm, sub_left_inj]
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝³ : CommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Module.IsReflexive R M
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      hNB : B.Nondegenerate
      hSB : B.IsSymm
      h2 : IsRegular 2
      x y : ↑(setOf fun x => B.IsReflective x)
      z : M
      ⊢ Eq (HMul.hMul (HMul.hMul 2 ((B ↑y) ↑x)) ((LinearMap.IsReflective.coroot B ⋯) …
    -/
    refine x.2.1.1 ?_
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝³ : CommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Module.IsReflexive R M
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      hNB : B.Nondegenerate
      hSB : B.IsSymm
      h2 : IsRegular 2
      x y : ↑(setOf fun x => B.IsReflective x)
      z : M
      ⊢ Eq ((fun x_1 => HMul.hMul ((B ↑x) ↑x) x_1) (HMul.hMul (HMul.hMul 2 ((B ↑y) ↑ …
    -/
    simp only [mem_setOf_eq, map_smul, smul_eq_mul]
    rw [← mul_assoc _ _ (B z x), ← mul_assoc _ _ (B z x), mul_left_comm,
      apply_self_mul_coroot_apply B x.2, mul_left_comm (B x x), apply_self_mul_coroot_apply B x.2,
      ← hSB x y, RingHom.id_apply, ← hSB x z, RingHom.id_apply]
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝³ : CommRing R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Module.IsReflexive R M
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      hNB : B.Nondegenerate
      hSB : B.IsSymm
      h2 : IsRegular 2
      x y : ↑(setOf fun x => B.IsReflective x)
      z : M
      ⊢ Eq (HMul.hMul (HMul.hMul 2 ((B ↑x) ↑y)) (HMul.hMul 2 ((B ↑x) z))) (HMul.hMul …
    -/
    ring
    /-
      🎉 no goals
    -/


