/-- A *Hilbert C⋆-module* is a complex module `E` endowed with a right `A`-module structure
(where `A` is typically a C⋆-algebra) and an inner product `⟪x, y⟫_A` which satisfies the
following properties. -/
class CStarModule (A : outParam <| Type*) (E : Type*) [NonUnitalSemiring A] [StarRing A]
    [Module ℂ A] [AddCommGroup E] [Module ℂ E] [PartialOrder A] [SMul Aᵐᵒᵖ E] [Norm A] [Norm E]
    extends Inner A E where
  inner_add_right {x} {y} {z} : inner x (y + z) = inner x y + inner x z
  inner_self_nonneg {x} : 0 ≤ inner x x
  inner_self {x} : inner x x = 0 ↔ x = 0
  inner_op_smul_right {a : A} {x y : E} : inner x (y <• a) = inner x y * a
  inner_smul_right_complex {z : ℂ} {x} {y} : inner x (z • y) = z • inner x y
  star_inner x y : star (inner x y) = inner y x
  norm_eq_sqrt_norm_inner_self x : ‖x‖ = √‖inner x x‖


@[deprecated (since := "2024-08-04")] alias CstarModule := CStarModule


local notation "⟪" x ", " y "⟫" => inner (𝕜 := A) x y


@[simp]
lemma inner_add_left {x y z : E} : ⟪x + y, z⟫ = ⟪x, z⟫ + ⟪y, z⟫ := by
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Complex A
    inst✝⁵ : Module Complex E
    inst✝⁴ : PartialOrder A
    inst✝³ : SMul (MulOpposite A) E
    inst✝² : Norm A
    inst✝¹ : Norm E
    inst✝ : CStarModule A E
    x y z : E
    ⊢ Eq (Inner.inner (HAdd.hAdd x y) z) (HAdd.hAdd (Inner.inner x z) (Inner.inner …
  -/
  rw [← star_star (r := ⟪x + y, z⟫)]
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Complex A
    inst✝⁵ : Module Complex E
    inst✝⁴ : PartialOrder A
    inst✝³ : SMul (MulOpposite A) E
    inst✝² : Norm A
    inst✝¹ : Norm E
    inst✝ : CStarModule A E
    x y z : E
    ⊢ Eq (Star.star (Star.star (Inner.inner (HAdd.hAdd x y) z))) (HAdd.hAdd (Inner …
  -/
  simp only [inner_add_right, star_add, star_inner]
  /-
    🎉 no goals
  -/


@[simp]
lemma inner_op_smul_left {a : A} {x y : E} : ⟪x <• a, y⟫ = star a * ⟪x, y⟫ := by
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁹ : NonUnitalRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Complex A
    inst✝⁵ : Module Complex E
    inst✝⁴ : PartialOrder A
    inst✝³ : SMul (MulOpposite A) E
    inst✝² : Norm A
    inst✝¹ : Norm E
    inst✝ : CStarModule A E
    a : A
    x y : E
    ⊢ Eq (Inner.inner (HSMul.hSMul (MulOpposite.op a) x) y) (HMul.hMul (Star.star  …
  -/
  rw [← star_inner]; simp
                     /-
                       🎉 no goals
                     -/


@[simp]
lemma inner_smul_left_complex {z : ℂ} {x y : E} : ⟪z • x, y⟫ = star z • ⟪x, y⟫ := by
  /-
    A : Type u_1
    E : Type u_2
    inst✝¹⁰ : NonUnitalRing A
    inst✝⁹ : StarRing A
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Complex A
    inst✝⁶ : Module Complex E
    inst✝⁵ : PartialOrder A
    inst✝⁴ : SMul (MulOpposite A) E
    inst✝³ : Norm A
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarModule Complex A
    z : Complex
    x y : E
    ⊢ Eq (Inner.inner (HSMul.hSMul z x) y) (HSMul.hSMul (Star.star z) (Inner.inner …
  -/
  rw [← star_inner]
  /-
    A : Type u_1
    E : Type u_2
    inst✝¹⁰ : NonUnitalRing A
    inst✝⁹ : StarRing A
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Complex A
    inst✝⁶ : Module Complex E
    inst✝⁵ : PartialOrder A
    inst✝⁴ : SMul (MulOpposite A) E
    inst✝³ : Norm A
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarModule Complex A
    z : Complex
    x y : E
    ⊢ Eq (Star.star (Inner.inner y (HSMul.hSMul z x))) (HSMul.hSMul (Star.star z)  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma inner_smul_left_real {z : ℝ} {x y : E} : ⟪z • x, y⟫ = z • ⟪x, y⟫ := by
  /-
    A : Type u_1
    E : Type u_2
    inst✝¹⁰ : NonUnitalRing A
    inst✝⁹ : StarRing A
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Complex A
    inst✝⁶ : Module Complex E
    inst✝⁵ : PartialOrder A
    inst✝⁴ : SMul (MulOpposite A) E
    inst✝³ : Norm A
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarModule Complex A
    z : Real
    x y : E
    ⊢ Eq (Inner.inner (HSMul.hSMul z x) y) (HSMul.hSMul z (Inner.inner x y))
  -/
  have h₁ : z • x = (z : ℂ) • x := by simp
  /-
    A : Type u_1
    E : Type u_2
    inst✝¹⁰ : NonUnitalRing A
    inst✝⁹ : StarRing A
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Complex A
    inst✝⁶ : Module Complex E
    inst✝⁵ : PartialOrder A
    inst✝⁴ : SMul (MulOpposite A) E
    inst✝³ : Norm A
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarModule Complex A
    z : Real
    x y : E
    h₁ : Eq (HSMul.hSMul z x) (HSMul.hSMul (↑z) x)
    ⊢ Eq (Inner.inner (HSMul.hSMul z x) y) (HSMul.hSMul z (Inner.inner x y))
  -/
  rw [h₁, ← star_inner, inner_smul_right_complex]
  /-
    A : Type u_1
    E : Type u_2
    inst✝¹⁰ : NonUnitalRing A
    inst✝⁹ : StarRing A
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Complex A
    inst✝⁶ : Module Complex E
    inst✝⁵ : PartialOrder A
    inst✝⁴ : SMul (MulOpposite A) E
    inst✝³ : Norm A
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarModule Complex A
    z : Real
    x y : E
    h₁ : Eq (HSMul.hSMul z x) (HSMul.hSMul (↑z) x)
    ⊢ Eq (Star.star (HSMul.hSMul (↑z) (Inner.inner y x))) (HSMul.hSMul z (Inner.in …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma inner_smul_right_real {z : ℝ} {x y : E} : ⟪x, z • y⟫ = z • ⟪x, y⟫ := by
  /-
    A : Type u_1
    E : Type u_2
    inst✝¹⁰ : NonUnitalRing A
    inst✝⁹ : StarRing A
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Complex A
    inst✝⁶ : Module Complex E
    inst✝⁵ : PartialOrder A
    inst✝⁴ : SMul (MulOpposite A) E
    inst✝³ : Norm A
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarModule Complex A
    z : Real
    x y : E
    ⊢ Eq (Inner.inner x (HSMul.hSMul z y)) (HSMul.hSMul z (Inner.inner x y))
  -/
  have h₁ : z • y = (z : ℂ) • y := by simp
  /-
    A : Type u_1
    E : Type u_2
    inst✝¹⁰ : NonUnitalRing A
    inst✝⁹ : StarRing A
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Complex A
    inst✝⁶ : Module Complex E
    inst✝⁵ : PartialOrder A
    inst✝⁴ : SMul (MulOpposite A) E
    inst✝³ : Norm A
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarModule Complex A
    z : Real
    x y : E
    h₁ : Eq (HSMul.hSMul z y) (HSMul.hSMul (↑z) y)
    ⊢ Eq (Inner.inner x (HSMul.hSMul z y)) (HSMul.hSMul z (Inner.inner x y))
  -/
  rw [h₁, ← star_inner, inner_smul_left_complex]
  /-
    A : Type u_1
    E : Type u_2
    inst✝¹⁰ : NonUnitalRing A
    inst✝⁹ : StarRing A
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Complex A
    inst✝⁶ : Module Complex E
    inst✝⁵ : PartialOrder A
    inst✝⁴ : SMul (MulOpposite A) E
    inst✝³ : Norm A
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarModule Complex A
    z : Real
    x y : E
    h₁ : Eq (HSMul.hSMul z y) (HSMul.hSMul (↑z) y)
    ⊢ Eq (Star.star (HSMul.hSMul (Star.star ↑z) (Inner.inner y x))) (HSMul.hSMul z …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The function `⟨x, y⟩ ↦ ⟪x, y⟫` bundled as a sesquilinear map. -/
def innerₛₗ : E →ₗ⋆[ℂ] E →ₗ[ℂ] A where
  toFun x := { toFun := fun y => ⟪x, y⟫
                                         /-
                                           A : Type u_1
                                           E : Type u_2
                                           inst✝¹⁰ : NonUnitalRing A
                                           inst✝⁹ : StarRing A
                                           inst✝⁸ : AddCommGroup E
                                           inst✝⁷ : Module Complex A
                                           inst✝⁶ : Module Complex E
                                           inst✝⁵ : PartialOrder A
                                           inst✝⁴ : SMul (MulOpposite A) E
                                           inst✝³ : Norm A
                                           inst✝² : Norm E
                                           inst✝¹ : CStarModule A E
                                           inst✝ : StarModule Complex A
                                           x z y : E
                                           ⊢ Eq ((fun y => Inner.inner x y) (HAdd.hAdd z y)) (HAdd.hAdd ((fun y => Inner. …
                                         -/
               map_add' := fun z y => by simp
                                         /-
                                           🎉 no goals
                                         -/
                                          /-
                                            A : Type u_1
                                            E : Type u_2
                                            inst✝¹⁰ : NonUnitalRing A
                                            inst✝⁹ : StarRing A
                                            inst✝⁸ : AddCommGroup E
                                            inst✝⁷ : Module Complex A
                                            inst✝⁶ : Module Complex E
                                            inst✝⁵ : PartialOrder A
                                            inst✝⁴ : SMul (MulOpposite A) E
                                            inst✝³ : Norm A
                                            inst✝² : Norm E
                                            inst✝¹ : CStarModule A E
                                            inst✝ : StarModule Complex A
                                            x : E
                                            z : Complex
                                            y : E
                                            ⊢ Eq ({ toFun := fun y => Inner.inner x y, map_add' := ⋯ }.toFun (HSMul.hSMul  …
                                          -/
               map_smul' := fun z y => by simp }
                                          /-
                                            🎉 no goals
                                          -/
                     /-
                       A : Type u_1
                       E : Type u_2
                       inst✝¹⁰ : NonUnitalRing A
                       inst✝⁹ : StarRing A
                       inst✝⁸ : AddCommGroup E
                       inst✝⁷ : Module Complex A
                       inst✝⁶ : Module Complex E
                       inst✝⁵ : PartialOrder A
                       inst✝⁴ : SMul (MulOpposite A) E
                       inst✝³ : Norm A
                       inst✝² : Norm E
                       inst✝¹ : CStarModule A E
                       inst✝ : StarModule Complex A
                       z y : E
                       ⊢ Eq ((fun x => { toFun := fun y => Inner.inner x y, map_add' := ⋯, map_smul'  …
                     -/
  map_add' z y := by ext; simp
                          /-
                            🎉 no goals
                          -/
                      /-
                        A : Type u_1
                        E : Type u_2
                        inst✝¹⁰ : NonUnitalRing A
                        inst✝⁹ : StarRing A
                        inst✝⁸ : AddCommGroup E
                        inst✝⁷ : Module Complex A
                        inst✝⁶ : Module Complex E
                        inst✝⁵ : PartialOrder A
                        inst✝⁴ : SMul (MulOpposite A) E
                        inst✝³ : Norm A
                        inst✝² : Norm E
                        inst✝¹ : CStarModule A E
                        inst✝ : StarModule Complex A
                        z : Complex
                        y : E
                        ⊢ Eq ({ toFun := fun x => { toFun := fun y => Inner.inner x y, map_add' := ⋯,  …
                      -/
  map_smul' z y := by ext; simp
                           /-
                             🎉 no goals
                           -/


lemma innerₛₗ_apply {x y : E} : innerₛₗ x y = ⟪x, y⟫ := rfl


                                                          /-
                                                            A : Type u_1
                                                            E : Type u_2
                                                            inst✝¹⁰ : NonUnitalRing A
                                                            inst✝⁹ : StarRing A
                                                            inst✝⁸ : AddCommGroup E
                                                            inst✝⁷ : Module Complex A
                                                            inst✝⁶ : Module Complex E
                                                            inst✝⁵ : PartialOrder A
                                                            inst✝⁴ : SMul (MulOpposite A) E
                                                            inst✝³ : Norm A
                                                            inst✝² : Norm E
                                                            inst✝¹ : CStarModule A E
                                                            inst✝ : StarModule Complex A
                                                            x : E
                                                            ⊢ Eq (Inner.inner x 0) 0
                                                          -/
@[simp] lemma inner_zero_right {x : E} : ⟪x, 0⟫ = 0 := by simp [← innerₛₗ_apply]
                                                          /-
                                                            🎉 no goals
                                                          -/

                                                         /-
                                                           A : Type u_1
                                                           E : Type u_2
                                                           inst✝¹⁰ : NonUnitalRing A
                                                           inst✝⁹ : StarRing A
                                                           inst✝⁸ : AddCommGroup E
                                                           inst✝⁷ : Module Complex A
                                                           inst✝⁶ : Module Complex E
                                                           inst✝⁵ : PartialOrder A
                                                           inst✝⁴ : SMul (MulOpposite A) E
                                                           inst✝³ : Norm A
                                                           inst✝² : Norm E
                                                           inst✝¹ : CStarModule A E
                                                           inst✝ : StarModule Complex A
                                                           x : E
                                                           ⊢ Eq (Inner.inner 0 x) 0
                                                         -/
@[simp] lemma inner_zero_left {x : E} : ⟪0, x⟫ = 0 := by simp [← innerₛₗ_apply]
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                                  /-
                                                                    A : Type u_1
                                                                    E : Type u_2
                                                                    inst✝¹⁰ : NonUnitalRing A
                                                                    inst✝⁹ : StarRing A
                                                                    inst✝⁸ : AddCommGroup E
                                                                    inst✝⁷ : Module Complex A
                                                                    inst✝⁶ : Module Complex E
                                                                    inst✝⁵ : PartialOrder A
                                                                    inst✝⁴ : SMul (MulOpposite A) E
                                                                    inst✝³ : Norm A
                                                                    inst✝² : Norm E
                                                                    inst✝¹ : CStarModule A E
                                                                    inst✝ : StarModule Complex A
                                                                    x y : E
                                                                    ⊢ Eq (Inner.inner x (Neg.neg y)) (Neg.neg (Inner.inner x y))
                                                                  -/
@[simp] lemma inner_neg_right {x y : E} : ⟪x, -y⟫ = -⟪x, y⟫ := by simp [← innerₛₗ_apply]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/

                                                                 /-
                                                                   A : Type u_1
                                                                   E : Type u_2
                                                                   inst✝¹⁰ : NonUnitalRing A
                                                                   inst✝⁹ : StarRing A
                                                                   inst✝⁸ : AddCommGroup E
                                                                   inst✝⁷ : Module Complex A
                                                                   inst✝⁶ : Module Complex E
                                                                   inst✝⁵ : PartialOrder A
                                                                   inst✝⁴ : SMul (MulOpposite A) E
                                                                   inst✝³ : Norm A
                                                                   inst✝² : Norm E
                                                                   inst✝¹ : CStarModule A E
                                                                   inst✝ : StarModule Complex A
                                                                   x y : E
                                                                   ⊢ Eq (Inner.inner (Neg.neg x) y) (Neg.neg (Inner.inner x y))
                                                                 -/
@[simp] lemma inner_neg_left {x y : E} : ⟪-x, y⟫ = -⟪x, y⟫ := by simp [← innerₛₗ_apply]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/

@[simp] lemma inner_sub_right {x y z : E} : ⟪x, y - z⟫ = ⟪x, y⟫ - ⟪x, z⟫ := by
  /-
    A : Type u_1
    E : Type u_2
    inst✝¹⁰ : NonUnitalRing A
    inst✝⁹ : StarRing A
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Complex A
    inst✝⁶ : Module Complex E
    inst✝⁵ : PartialOrder A
    inst✝⁴ : SMul (MulOpposite A) E
    inst✝³ : Norm A
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarModule Complex A
    x y z : E
    ⊢ Eq (Inner.inner x (HSub.hSub y z)) (HSub.hSub (Inner.inner x y) (Inner.inner …
  -/
  simp [← innerₛₗ_apply]
  /-
    🎉 no goals
  -/

@[simp] lemma inner_sub_left {x y z : E} : ⟪x - y, z⟫ = ⟪x, z⟫ - ⟪y, z⟫ := by
  /-
    A : Type u_1
    E : Type u_2
    inst✝¹⁰ : NonUnitalRing A
    inst✝⁹ : StarRing A
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Complex A
    inst✝⁶ : Module Complex E
    inst✝⁵ : PartialOrder A
    inst✝⁴ : SMul (MulOpposite A) E
    inst✝³ : Norm A
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarModule Complex A
    x y z : E
    ⊢ Eq (Inner.inner (HSub.hSub x y) z) (HSub.hSub (Inner.inner x z) (Inner.inner …
  -/
  simp [← innerₛₗ_apply]
  /-
    🎉 no goals
  -/


@[simp]
lemma inner_sum_right {ι : Type*} {s : Finset ι} {x : E} {y : ι → E} :
    ⟪x, ∑ i ∈ s, y i⟫ = ∑ i ∈ s, ⟪x, y i⟫ :=
  map_sum (innerₛₗ x) ..


@[simp]
lemma inner_sum_left {ι : Type*} {s : Finset ι} {x : ι → E} {y : E} :
    ⟪∑ i ∈ s, x i, y⟫ = ∑ i ∈ s, ⟪x i, y⟫ :=
  map_sum (innerₛₗ.flip y) ..


@[simp]
lemma isSelfAdjoint_inner_self {x : E} : IsSelfAdjoint ⟪x, x⟫ := star_inner _ _


open scoped InnerProductSpace in
/-- The norm associated with a Hilbert C⋆-module. It is not registered as a norm, since a type
might already have a norm defined on it. -/
noncomputable def norm (A : Type*) {E : Type*} [Norm A] [Inner A E] : Norm E where
  norm x := Real.sqrt ‖⟪x, x⟫_A‖


                                                    /-
                                                      A : Type u_1
                                                      E : Type u_2
                                                      inst✝⁶ : NonUnitalCStarAlgebra A
                                                      inst✝⁵ : PartialOrder A
                                                      inst✝⁴ : AddCommGroup E
                                                      inst✝³ : Module Complex E
                                                      inst✝² : SMul (MulOpposite A) E
                                                      inst✝¹ : Norm E
                                                      inst✝ : CStarModule A E
                                                      x : E
                                                      ⊢ Eq (HPow.hPow (Norm.norm x) 2) (Norm.norm (Inner.inner x x))
                                                    -/
lemma norm_sq_eq {x : E} : ‖x‖ ^ 2 = ‖⟪x, x⟫‖ := by simp [norm_eq_sqrt_norm_inner_self]
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                    /-
                                                      A : Type u_1
                                                      E : Type u_2
                                                      inst✝⁶ : NonUnitalCStarAlgebra A
                                                      inst✝⁵ : PartialOrder A
                                                      inst✝⁴ : AddCommGroup E
                                                      inst✝³ : Module Complex E
                                                      inst✝² : SMul (MulOpposite A) E
                                                      inst✝¹ : Norm E
                                                      inst✝ : CStarModule A E
                                                      x : E
                                                      ⊢ LE.le 0 (Norm.norm x)
                                                    -/
protected lemma norm_nonneg {x : E} : 0 ≤ ‖x‖ := by simp [norm_eq_sqrt_norm_inner_self]
                                                    /-
                                                      🎉 no goals
                                                    -/


protected lemma norm_pos {x : E} (hx : x ≠ 0) : 0 < ‖x‖ := by
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁶ : NonUnitalCStarAlgebra A
    inst✝⁵ : PartialOrder A
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Complex E
    inst✝² : SMul (MulOpposite A) E
    inst✝¹ : Norm E
    inst✝ : CStarModule A E
    x : E
    hx : Ne x 0
    ⊢ LT.lt 0 (Norm.norm x)
  -/
  simp only [norm_eq_sqrt_norm_inner_self, Real.sqrt_pos, norm_pos_iff]
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁶ : NonUnitalCStarAlgebra A
    inst✝⁵ : PartialOrder A
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Complex E
    inst✝² : SMul (MulOpposite A) E
    inst✝¹ : Norm E
    inst✝ : CStarModule A E
    x : E
    hx : Ne x 0
    ⊢ Ne (Inner.inner x x) 0
  -/
  intro H
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁶ : NonUnitalCStarAlgebra A
    inst✝⁵ : PartialOrder A
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Complex E
    inst✝² : SMul (MulOpposite A) E
    inst✝¹ : Norm E
    inst✝ : CStarModule A E
    x : E
    hx : Ne x 0
    H : Eq (Inner.inner x x) 0
    ⊢ False
  -/
  rw [inner_self] at H
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁶ : NonUnitalCStarAlgebra A
    inst✝⁵ : PartialOrder A
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Complex E
    inst✝² : SMul (MulOpposite A) E
    inst✝¹ : Norm E
    inst✝ : CStarModule A E
    x : E
    hx : Ne x 0
    H : Eq x 0
    ⊢ False
  -/
  exact hx H
  /-
    🎉 no goals
  -/


                                                /-
                                                  A : Type u_1
                                                  E : Type u_2
                                                  inst✝⁶ : NonUnitalCStarAlgebra A
                                                  inst✝⁵ : PartialOrder A
                                                  inst✝⁴ : AddCommGroup E
                                                  inst✝³ : Module Complex E
                                                  inst✝² : SMul (MulOpposite A) E
                                                  inst✝¹ : Norm E
                                                  inst✝ : CStarModule A E
                                                  ⊢ Eq (Norm.norm 0) 0
                                                -/
protected lemma norm_zero : ‖(0 : E)‖ = 0 := by simp [norm_eq_sqrt_norm_inner_self]
                                                /-
                                                  🎉 no goals
                                                -/


lemma norm_zero_iff (x : E) : ‖x‖ = 0 ↔ x = 0 :=
               /-
                 A : Type u_1
                 E : Type u_2
                 inst✝⁶ : NonUnitalCStarAlgebra A
                 inst✝⁵ : PartialOrder A
                 inst✝⁴ : AddCommGroup E
                 inst✝³ : Module Complex E
                 inst✝² : SMul (MulOpposite A) E
                 inst✝¹ : Norm E
                 inst✝ : CStarModule A E
                 x : E
                 h : Eq (Norm.norm x) 0
                 ⊢ Eq x 0
               -/
  ⟨fun h => by simpa [norm_eq_sqrt_norm_inner_self, inner_self] using h,
               /-
                 🎉 no goals
               -/
                /-
                  A : Type u_1
                  E : Type u_2
                  inst✝⁶ : NonUnitalCStarAlgebra A
                  inst✝⁵ : PartialOrder A
                  inst✝⁴ : AddCommGroup E
                  inst✝³ : Module Complex E
                  inst✝² : SMul (MulOpposite A) E
                  inst✝¹ : Norm E
                  inst✝ : CStarModule A E
                  x : E
                  h : Eq x 0
                  ⊢ Eq (Norm.norm x) 0
                -/
    fun h => by simp [norm, h, norm_eq_sqrt_norm_inner_self]⟩
                /-
                  🎉 no goals
                -/


open scoped InnerProductSpace in
/-- The C⋆-algebra-valued Cauchy-Schwarz inequality for Hilbert C⋆-modules. -/
lemma inner_mul_inner_swap_le {x y : E} : ⟪y, x⟫ * ⟪x, y⟫ ≤ ‖x‖ ^ 2 • ⟪y, y⟫ := by
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁷ : NonUnitalCStarAlgebra A
    inst✝⁶ : PartialOrder A
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Complex E
    inst✝³ : SMul (MulOpposite A) E
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarOrderedRing A
    x y : E
    ⊢ LE.le (HMul.hMul (Inner.inner y x) (Inner.inner x y)) (HSMul.hSMul (HPow.hPo …
  -/
  rcases eq_or_ne x 0 with h|h
    /-
      case inl
      A : Type u_1
      E : Type u_2
      inst✝⁷ : NonUnitalCStarAlgebra A
      inst✝⁶ : PartialOrder A
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module Complex E
      inst✝³ : SMul (MulOpposite A) E
      inst✝² : Norm E
      inst✝¹ : CStarModule A E
      inst✝ : StarOrderedRing A
      x y : E
      h : Eq x 0
      ⊢ LE.le (HMul.hMul (Inner.inner y x) (Inner.inner x y)) (HSMul.hSMul (HPow.hPo …
    -/
  · simp [h, CStarModule.norm_zero (E := E)]
    /-
      🎉 no goals
    -/
  · have h₁ : ∀ (a : A),
        (0 : A) ≤ ‖x‖ ^ 2 • (star a * a) - ‖x‖ ^ 2 • (⟪y, x⟫ * a)
                  - ‖x‖ ^ 2 • (star a * ⟪x, y⟫) + ‖x‖ ^ 2 • (‖x‖ ^ 2 • ⟪y, y⟫) := fun a => by
      calc (0 : A) ≤ ⟪x <• a - ‖x‖ ^ 2 • y, x <• a - ‖x‖ ^ 2 • y⟫_A := by
                      exact inner_self_nonneg
            _ = star a * ⟪x, x⟫ * a - ‖x‖ ^ 2 • (⟪y, x⟫ * a)
                  - ‖x‖ ^ 2 • (star a * ⟪x, y⟫) + ‖x‖ ^ 2 • (‖x‖ ^ 2 • ⟪y, y⟫) := by
                      simp only [inner_sub_right, inner_op_smul_right, inner_sub_left,
                        inner_op_smul_left, inner_smul_left_real, sub_mul, smul_mul_assoc,
                        inner_smul_right_real, smul_sub]
                      abel
            _ ≤ ‖x‖ ^ 2 • (star a * a) - ‖x‖ ^ 2 • (⟪y, x⟫ * a)
                  - ‖x‖ ^ 2 • (star a * ⟪x, y⟫) + ‖x‖ ^ 2 • (‖x‖ ^ 2 • ⟪y, y⟫) := by
                      gcongr
                      calc _ ≤ ‖⟪x, x⟫_A‖ • (star a * a) := CStarAlgebra.conjugate_le_norm_smul
                        _ = (Real.sqrt ‖⟪x, x⟫_A‖) ^ 2 • (star a * a) := by
                                  congr
                                  have : 0 ≤ ‖⟪x, x⟫_A‖ := by positivity
                                  rw [Real.sq_sqrt this]
                        _ = ‖x‖ ^ 2 • (star a * a) := by rw [← norm_eq_sqrt_norm_inner_self]
    /-
      case inr
      A : Type u_1
      E : Type u_2
      inst✝⁷ : NonUnitalCStarAlgebra A
      inst✝⁶ : PartialOrder A
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module Complex E
      inst✝³ : SMul (MulOpposite A) E
      inst✝² : Norm E
      inst✝¹ : CStarModule A E
      inst✝ : StarOrderedRing A
      x y : E
      h : Ne x 0
      h₁ : ∀ (a : A), LE.le 0 (HAdd.hAdd (HSub.hSub (HSub.hSub (HSMul.hSMul (HPow.hP …
      ⊢ LE.le (HMul.hMul (Inner.inner y x) (Inner.inner x y)) (HSMul.hSMul (HPow.hPo …
    -/
    specialize h₁ ⟪x, y⟫
    /-
      case inr
      A : Type u_1
      E : Type u_2
      inst✝⁷ : NonUnitalCStarAlgebra A
      inst✝⁶ : PartialOrder A
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module Complex E
      inst✝³ : SMul (MulOpposite A) E
      inst✝² : Norm E
      inst✝¹ : CStarModule A E
      inst✝ : StarOrderedRing A
      x y : E
      h : Ne x 0
      h₁ : LE.le 0 (HAdd.hAdd (HSub.hSub (HSub.hSub (HSMul.hSMul (HPow.hPow (Norm.no …
      ⊢ LE.le (HMul.hMul (Inner.inner y x) (Inner.inner x y)) (HSMul.hSMul (HPow.hPo …
    -/
    simp only [star_inner, sub_self, zero_sub, le_neg_add_iff_add_le, add_zero] at h₁
    /-
      case inr
      A : Type u_1
      E : Type u_2
      inst✝⁷ : NonUnitalCStarAlgebra A
      inst✝⁶ : PartialOrder A
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module Complex E
      inst✝³ : SMul (MulOpposite A) E
      inst✝² : Norm E
      inst✝¹ : CStarModule A E
      inst✝ : StarOrderedRing A
      x y : E
      h : Ne x 0
      h₁ : LE.le (HSMul.hSMul (HPow.hPow (Norm.norm x) 2) (HMul.hMul (Inner.inner y  …
      ⊢ LE.le (HMul.hMul (Inner.inner y x) (Inner.inner x y)) (HSMul.hSMul (HPow.hPo …
    -/
    rwa [smul_le_smul_iff_of_pos_left (pow_pos (CStarModule.norm_pos h) _)] at h₁
    /-
      🎉 no goals
    -/


open scoped InnerProductSpace in
variable (E) in
/-- The Cauchy-Schwarz inequality for Hilbert C⋆-modules. -/
lemma norm_inner_le {x y : E} : ‖⟪x, y⟫‖ ≤ ‖x‖ * ‖y‖ := by
  have := calc ‖⟪x, y⟫‖ ^ 2 = ‖⟪y, x⟫ * ⟪x, y⟫‖ := by
                rw [← star_inner x, CStarRing.norm_star_mul_self, pow_two]
    _ ≤ ‖‖x‖^ 2 • ⟪y, y⟫‖ := by
                refine CStarAlgebra.norm_le_norm_of_nonneg_of_le ?_ inner_mul_inner_swap_le
                rw [← star_inner x]
                exact star_mul_self_nonneg ⟪x, y⟫_A
    _ = ‖x‖ ^ 2 * ‖⟪y, y⟫‖ := by simp [norm_smul]
    _ = ‖x‖ ^ 2 * ‖y‖ ^ 2 := by
                simp only [norm_eq_sqrt_norm_inner_self, norm_nonneg, Real.sq_sqrt]
    _ = (‖x‖ * ‖y‖) ^ 2 := by simp only [mul_pow]
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁷ : NonUnitalCStarAlgebra A
    inst✝⁶ : PartialOrder A
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Complex E
    inst✝³ : SMul (MulOpposite A) E
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarOrderedRing A
    x y : E
    this : LE.le (HPow.hPow (Norm.norm (Inner.inner x y)) 2) (HPow.hPow (HMul.hMul …
    ⊢ LE.le (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
  -/
  refine (pow_le_pow_iff_left₀ (norm_nonneg ⟪x, y⟫_A) ?_ (by norm_num)).mp this
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁷ : NonUnitalCStarAlgebra A
    inst✝⁶ : PartialOrder A
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Complex E
    inst✝³ : SMul (MulOpposite A) E
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarOrderedRing A
    x y : E
    this : LE.le (HPow.hPow (Norm.norm (Inner.inner x y)) 2) (HPow.hPow (HMul.hMul …
    ⊢ LE.le 0 (HMul.hMul (Norm.norm x) (Norm.norm y))
  -/
  exact mul_nonneg CStarModule.norm_nonneg CStarModule.norm_nonneg
  /-
    🎉 no goals
  -/


include A in
protected lemma norm_triangle (x y : E) : ‖x + y‖ ≤ ‖x‖ + ‖y‖ := by
  have h : ‖x + y‖ ^ 2 ≤ (‖x‖ + ‖y‖) ^ 2 := by
    calc _ ≤ ‖⟪x, x⟫ + ⟪y, x⟫‖ + ‖⟪x, y⟫‖ + ‖⟪y, y⟫‖ := by
          simp only [norm_eq_sqrt_norm_inner_self, inner_add_right, inner_add_left, ← add_assoc,
            norm_nonneg, Real.sq_sqrt]
          exact norm_add₃_le
      _ ≤ ‖⟪x, x⟫‖ + ‖⟪y, x⟫‖ + ‖⟪x, y⟫‖ + ‖⟪y, y⟫‖ := by gcongr; exact norm_add_le _ _
      _ ≤ ‖⟪x, x⟫‖ + ‖y‖ * ‖x‖ + ‖x‖ * ‖y‖ + ‖⟪y, y⟫‖ := by gcongr <;> exact norm_inner_le E
      _ = ‖x‖ ^ 2 + ‖y‖ * ‖x‖ + ‖x‖ * ‖y‖ + ‖y‖ ^ 2 := by
          simp [norm_eq_sqrt_norm_inner_self]
      _ = (‖x‖ + ‖y‖) ^ 2 := by simp only [add_pow_two, add_left_inj]; ring
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁷ : NonUnitalCStarAlgebra A
    inst✝⁶ : PartialOrder A
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Complex E
    inst✝³ : SMul (MulOpposite A) E
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarOrderedRing A
    x y : E
    h : LE.le (HPow.hPow (Norm.norm (HAdd.hAdd x y)) 2) (HPow.hPow (HAdd.hAdd (Nor …
    ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
  -/
  refine (pow_le_pow_iff_left₀ CStarModule.norm_nonneg ?_ (by norm_num)).mp h
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁷ : NonUnitalCStarAlgebra A
    inst✝⁶ : PartialOrder A
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Complex E
    inst✝³ : SMul (MulOpposite A) E
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarOrderedRing A
    x y : E
    h : LE.le (HPow.hPow (Norm.norm (HAdd.hAdd x y)) 2) (HPow.hPow (HAdd.hAdd (Nor …
    ⊢ LE.le 0 (HAdd.hAdd (Norm.norm x) (Norm.norm y))
  -/
  exact add_nonneg CStarModule.norm_nonneg CStarModule.norm_nonneg
  /-
    🎉 no goals
  -/


include A in
/-- This allows us to get `NormedAddCommGroup` and `NormedSpace` instances on `E` via
`NormedAddCommGroup.ofCore` and `NormedSpace.ofCore`. -/
lemma normedSpaceCore : NormedSpace.Core ℂ E where
  norm_nonneg _ := CStarModule.norm_nonneg
  norm_eq_zero_iff x := norm_zero_iff x
                      /-
                        A : Type u_1
                        E : Type u_2
                        inst✝⁷ : NonUnitalCStarAlgebra A
                        inst✝⁶ : PartialOrder A
                        inst✝⁵ : AddCommGroup E
                        inst✝⁴ : Module Complex E
                        inst✝³ : SMul (MulOpposite A) E
                        inst✝² : Norm E
                        inst✝¹ : CStarModule A E
                        inst✝ : StarOrderedRing A
                        c : Complex
                        x : E
                        ⊢ Eq (Norm.norm (HSMul.hSMul c x)) (HMul.hMul (Norm.norm c) (Norm.norm x))
                      -/
  norm_smul c x := by simp [norm_eq_sqrt_norm_inner_self, norm_smul, ← mul_assoc]
                      /-
                        🎉 no goals
                      -/
  norm_triangle x y := CStarModule.norm_triangle x y


/-- This is not listed as an instance because we often want to replace the topology, uniformity
and bornology instead of inheriting them from the norm. -/
abbrev normedAddCommGroup : NormedAddCommGroup E :=
  NormedAddCommGroup.ofCore CStarModule.normedSpaceCore


open scoped InnerProductSpace in
lemma norm_eq_csSup (v : E) :
    ‖v‖ = sSup { ‖⟪w, v⟫_A‖ | (w : E) (_ : ‖w‖ ≤ 1) } := by
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁷ : NonUnitalCStarAlgebra A
    inst✝⁶ : PartialOrder A
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Complex E
    inst✝³ : SMul (MulOpposite A) E
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarOrderedRing A
    v : E
    ⊢ Eq (Norm.norm v) (SupSet.sSup (setOf fun x => Exists fun w => Exists fun x_1 …
  -/
  let instNACG : NormedAddCommGroup E := NormedAddCommGroup.ofCore normedSpaceCore
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁷ : NonUnitalCStarAlgebra A
    inst✝⁶ : PartialOrder A
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Complex E
    inst✝³ : SMul (MulOpposite A) E
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarOrderedRing A
    v : E
    instNACG : NormedAddCommGroup E := NormedAddCommGroup.ofCore ⋯
    ⊢ Eq (Norm.norm v) (SupSet.sSup (setOf fun x => Exists fun w => Exists fun x_1 …
  -/
  let instNS : NormedSpace ℂ E := .ofCore normedSpaceCore
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁷ : NonUnitalCStarAlgebra A
    inst✝⁶ : PartialOrder A
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Complex E
    inst✝³ : SMul (MulOpposite A) E
    inst✝² : Norm E
    inst✝¹ : CStarModule A E
    inst✝ : StarOrderedRing A
    v : E
    instNACG : NormedAddCommGroup E := NormedAddCommGroup.ofCore ⋯
    instNS : NormedSpace Complex E := NormedSpace.ofCore ⋯
    ⊢ Eq (Norm.norm v) (SupSet.sSup (setOf fun x => Exists fun w => Exists fun x_1 …
  -/
  refine Eq.symm <| IsGreatest.csSup_eq ⟨⟨‖v‖⁻¹ • v, ?_, ?_⟩, ?_⟩
    /-
      case refine_1
      A : Type u_1
      E : Type u_2
      inst✝⁷ : NonUnitalCStarAlgebra A
      inst✝⁶ : PartialOrder A
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module Complex E
      inst✝³ : SMul (MulOpposite A) E
      inst✝² : Norm E
      inst✝¹ : CStarModule A E
      inst✝ : StarOrderedRing A
      v : E
      instNACG : NormedAddCommGroup E := NormedAddCommGroup.ofCore ⋯
      instNS : NormedSpace Complex E := NormedSpace.ofCore ⋯
      ⊢ LE.le (Norm.norm (HSMul.hSMul (Inv.inv (Norm.norm v)) v)) 1
    -/
  · simpa only [norm_smul, norm_inv, norm_norm] using inv_mul_le_one_of_le₀ le_rfl (by positivity)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      A : Type u_1
      E : Type u_2
      inst✝⁷ : NonUnitalCStarAlgebra A
      inst✝⁶ : PartialOrder A
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module Complex E
      inst✝³ : SMul (MulOpposite A) E
      inst✝² : Norm E
      inst✝¹ : CStarModule A E
      inst✝ : StarOrderedRing A
      v : E
      instNACG : NormedAddCommGroup E := NormedAddCommGroup.ofCore ⋯
      instNS : NormedSpace Complex E := NormedSpace.ofCore ⋯
      ⊢ Eq (Norm.norm (Inner.inner (HSMul.hSMul (Inv.inv (Norm.norm v)) v) v)) (Norm …
    -/
  · simp [norm_smul, ← norm_sq_eq, pow_two, ← mul_assoc]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      A : Type u_1
      E : Type u_2
      inst✝⁷ : NonUnitalCStarAlgebra A
      inst✝⁶ : PartialOrder A
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module Complex E
      inst✝³ : SMul (MulOpposite A) E
      inst✝² : Norm E
      inst✝¹ : CStarModule A E
      inst✝ : StarOrderedRing A
      v : E
      instNACG : NormedAddCommGroup E := NormedAddCommGroup.ofCore ⋯
      instNS : NormedSpace Complex E := NormedSpace.ofCore ⋯
      ⊢ Membership.mem (upperBounds (setOf fun x => Exists fun w => Exists fun x_1 = …
    -/
  · rintro - ⟨w, hw, rfl⟩
    calc _ ≤ ‖w‖ * ‖v‖ := norm_inner_le E
      _ ≤ 1 * ‖v‖ := by gcongr
      _ = ‖v‖ := by simp


/-- The function `⟨x, y⟩ ↦ ⟪x, y⟫` bundled as a continuous sesquilinear map. -/
noncomputable def innerSL : E →L⋆[ℂ] E →L[ℂ] A :=
  LinearMap.mkContinuous₂ (innerₛₗ : E →ₗ⋆[ℂ] E →ₗ[ℂ] A) 1 <| fun x y => by
    /-
      A : Type u_1
      E : Type u_2
      inst✝⁶ : NonUnitalCStarAlgebra A
      inst✝⁵ : PartialOrder A
      inst✝⁴ : StarOrderedRing A
      inst✝³ : SMul (MulOpposite A) E
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      inst✝ : CStarModule A E
      x y : E
      ⊢ LE.le (Norm.norm ((CStarModule.innerₛₗ x) y)) (HMul.hMul (HMul.hMul 1 (Norm. …
    -/
    simp [innerₛₗ_apply, norm_inner_le E]
    /-
      🎉 no goals
    -/


lemma innerSL_apply {x y : E} : innerSL x y = ⟪x, y⟫_A := rfl


@[continuity, fun_prop]
lemma continuous_inner : Continuous (fun x : E × E => ⟪x.1, x.2⟫_A) := by
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁶ : NonUnitalCStarAlgebra A
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : SMul (MulOpposite A) E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CStarModule A E
    ⊢ Continuous fun x => Inner.inner x.1 x.2
  -/
  simp_rw [← innerSL_apply]
  /-
    A : Type u_1
    E : Type u_2
    inst✝⁶ : NonUnitalCStarAlgebra A
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : SMul (MulOpposite A) E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CStarModule A E
    ⊢ Continuous fun x => (CStarModule.innerSL x.1) x.2
  -/
  fun_prop
  /-
    🎉 no goals
  -/


