/-- The cross product of two vectors in $R^3$ for $R$ a commutative ring. -/
def crossProduct : (Fin 3 → R) →ₗ[R] (Fin 3 → R) →ₗ[R] Fin 3 → R := by
  apply LinearMap.mk₂ R fun a b : Fin 3 → R =>
      ![a 1 * b 2 - a 2 * b 1, a 2 * b 0 - a 0 * b 2, a 0 * b 1 - a 1 * b 0]
    /-
      case H1
      R : Type u_1
      inst✝ : CommRing R
      ⊢ ∀ (m₁ m₂ n : Fin 3 → R), Eq (Matrix.vecCons (HSub.hSub (HMul.hMul (HAdd.hAdd …
    -/
  · intros
    /-
      case H1
      R : Type u_1
      inst✝ : CommRing R
      m₁✝ m₂✝ n✝ : Fin 3 → R
      ⊢ Eq (Matrix.vecCons (HSub.hSub (HMul.hMul (HAdd.hAdd m₁✝ m₂✝ 1) (n✝ 2)) (HMul …
    -/
    simp_rw [vec3_add, Pi.add_apply]
    /-
      case H1
      R : Type u_1
      inst✝ : CommRing R
      m₁✝ m₂✝ n✝ : Fin 3 → R
      ⊢ Eq (Matrix.vecCons (HSub.hSub (HMul.hMul (HAdd.hAdd (m₁✝ 1) (m₂✝ 1)) (n✝ 2)) …
    -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
    apply vec3_eq <;> ring
                      /-
                        🎉 no goals
                      -/
    /-
      case H2
      R : Type u_1
      inst✝ : CommRing R
      ⊢ ∀ (c : R) (m n : Fin 3 → R), Eq (Matrix.vecCons (HSub.hSub (HMul.hMul (HSMul …
    -/
  · intros
    /-
      case H2
      R : Type u_1
      inst✝ : CommRing R
      c✝ : R
      m✝ n✝ : Fin 3 → R
      ⊢ Eq (Matrix.vecCons (HSub.hSub (HMul.hMul (HSMul.hSMul c✝ m✝ 1) (n✝ 2)) (HMul …
    -/
    simp_rw [smul_vec3, Pi.smul_apply, smul_sub, smul_mul_assoc]
    /-
      🎉 no goals
    -/
    /-
      case H3
      R : Type u_1
      inst✝ : CommRing R
      ⊢ ∀ (m n₁ n₂ : Fin 3 → R), Eq (Matrix.vecCons (HSub.hSub (HMul.hMul (m 1) (HAd …
    -/
  · intros
    /-
      case H3
      R : Type u_1
      inst✝ : CommRing R
      m✝ n₁✝ n₂✝ : Fin 3 → R
      ⊢ Eq (Matrix.vecCons (HSub.hSub (HMul.hMul (m✝ 1) (HAdd.hAdd n₁✝ n₂✝ 2)) (HMul …
    -/
    simp_rw [vec3_add, Pi.add_apply]
    /-
      case H3
      R : Type u_1
      inst✝ : CommRing R
      m✝ n₁✝ n₂✝ : Fin 3 → R
      ⊢ Eq (Matrix.vecCons (HSub.hSub (HMul.hMul (m✝ 1) (HAdd.hAdd (n₁✝ 2) (n₂✝ 2))) …
    -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
    apply vec3_eq <;> ring
                      /-
                        🎉 no goals
                      -/
    /-
      case H4
      R : Type u_1
      inst✝ : CommRing R
      ⊢ ∀ (c : R) (m n : Fin 3 → R), Eq (Matrix.vecCons (HSub.hSub (HMul.hMul (m 1)  …
    -/
  · intros
    /-
      case H4
      R : Type u_1
      inst✝ : CommRing R
      c✝ : R
      m✝ n✝ : Fin 3 → R
      ⊢ Eq (Matrix.vecCons (HSub.hSub (HMul.hMul (m✝ 1) (HSMul.hSMul c✝ n✝ 2)) (HMul …
    -/
    simp_rw [smul_vec3, Pi.smul_apply, smul_sub, mul_smul_comm]
    /-
      🎉 no goals
    -/


@[inherit_doc] scoped[Matrix] infixl:74 " ×₃ " => crossProduct


theorem cross_apply (a b : Fin 3 → R) :
    a ×₃ b = ![a 1 * b 2 - a 2 * b 1, a 2 * b 0 - a 0 * b 2, a 0 * b 1 - a 1 * b 0] := rfl


@[simp]
theorem cross_anticomm (v w : Fin 3 → R) : -(v ×₃ w) = w ×₃ v := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    v w : Fin 3 → R
    ⊢ Eq (Neg.neg ((crossProduct v) w)) ((crossProduct w) v)
  -/
  simp [cross_apply, mul_comm]
  /-
    🎉 no goals
  -/


alias neg_cross := cross_anticomm


@[simp]
theorem cross_anticomm' (v w : Fin 3 → R) : v ×₃ w + w ×₃ v = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    v w : Fin 3 → R
    ⊢ Eq (HAdd.hAdd ((crossProduct v) w) ((crossProduct w) v)) 0
  -/
  rw [add_eq_zero_iff_eq_neg, cross_anticomm]
  /-
    🎉 no goals
  -/


@[simp]
theorem cross_self (v : Fin 3 → R) : v ×₃ v = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    v : Fin 3 → R
    ⊢ Eq ((crossProduct v) v) 0
  -/
  simp [cross_apply, mul_comm]
  /-
    🎉 no goals
  -/


/-- The cross product of two vectors is perpendicular to the first vector. -/
@[simp 1100] -- Porting note: increase priority so that the LHS doesn't simplify
theorem dot_self_cross (v w : Fin 3 → R) : v ⬝ᵥ v ×₃ w = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    v w : Fin 3 → R
    ⊢ Eq (dotProduct v ((crossProduct v) w)) 0
  -/
  rw [cross_apply, vec3_dotProduct]
  /-
    R : Type u_1
    inst✝ : CommRing R
    v w : Fin 3 → R
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (v 0) (Matrix.vecCons (HSub.hSub (HMul.h …
  -/
  norm_num
  /-
    R : Type u_1
    inst✝ : CommRing R
    v w : Fin 3 → R
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (v 0) (HSub.hSub (HMul.hMul (v 1) (w 2)) …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- The cross product of two vectors is perpendicular to the second vector. -/
@[simp 1100] -- Porting note: increase priority so that the LHS doesn't simplify
theorem dot_cross_self (v w : Fin 3 → R) : w ⬝ᵥ v ×₃ w = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    v w : Fin 3 → R
    ⊢ Eq (dotProduct w ((crossProduct v) w)) 0
  -/
  rw [← cross_anticomm, dotProduct_neg, dot_self_cross, neg_zero]
  /-
    🎉 no goals
  -/


/-- Cyclic permutations preserve the triple product. See also `triple_product_eq_det`. -/
theorem triple_product_permutation (u v w : Fin 3 → R) : u ⬝ᵥ v ×₃ w = v ⬝ᵥ w ×₃ u := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    u v w : Fin 3 → R
    ⊢ Eq (dotProduct u ((crossProduct v) w)) (dotProduct v ((crossProduct w) u))
  -/
  simp_rw [cross_apply, vec3_dotProduct]
  /-
    R : Type u_1
    inst✝ : CommRing R
    u v w : Fin 3 → R
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (u 0) (Matrix.vecCons (HSub.hSub (HMul.h …
  -/
  norm_num
  /-
    R : Type u_1
    inst✝ : CommRing R
    u v w : Fin 3 → R
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (u 0) (HSub.hSub (HMul.hMul (v 1) (w 2)) …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- The triple product of `u`, `v`, and `w` is equal to the determinant of the matrix
    with those vectors as its rows. -/
theorem triple_product_eq_det (u v w : Fin 3 → R) : u ⬝ᵥ v ×₃ w = Matrix.det ![u, v, w] := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    u v w : Fin 3 → R
    ⊢ Eq (dotProduct u ((crossProduct v) w)) (Matrix.det (Matrix.vecCons u (Matrix …
  -/
  rw [vec3_dotProduct, cross_apply, det_fin_three]
  /-
    R : Type u_1
    inst✝ : CommRing R
    u v w : Fin 3 → R
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (u 0) (Matrix.vecCons (HSub.hSub (HMul.h …
  -/
  norm_num
  /-
    R : Type u_1
    inst✝ : CommRing R
    u v w : Fin 3 → R
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (u 0) (HSub.hSub (HMul.hMul (v 1) (w 2)) …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- The scalar quadruple product identity, related to the Binet-Cauchy identity. -/
theorem cross_dot_cross (u v w x : Fin 3 → R) :
    u ×₃ v ⬝ᵥ w ×₃ x = u ⬝ᵥ w * v ⬝ᵥ x - u ⬝ᵥ x * v ⬝ᵥ w := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    u v w x : Fin 3 → R
    ⊢ Eq (dotProduct ((crossProduct u) v) ((crossProduct w) x)) (HSub.hSub (HMul.h …
  -/
  simp_rw [cross_apply, vec3_dotProduct]
  /-
    R : Type u_1
    inst✝ : CommRing R
    u v w x : Fin 3 → R
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Matrix.vecCons (HSub.hSub (HMul.hMul (u …
  -/
  norm_num
  /-
    R : Type u_1
    inst✝ : CommRing R
    u v w x : Fin 3 → R
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HSub.hSub (HMul.hMul (u 1) (v 2)) (HMul …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- The cross product satisfies the Leibniz lie property. -/
theorem leibniz_cross (u v w : Fin 3 → R) : u ×₃ (v ×₃ w) = u ×₃ v ×₃ w + v ×₃ (u ×₃ w) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    u v w : Fin 3 → R
    ⊢ Eq ((crossProduct u) ((crossProduct v) w)) (HAdd.hAdd ((crossProduct ((cross …
  -/
  simp_rw [cross_apply, vec3_add]
  /-
    R : Type u_1
    inst✝ : CommRing R
    u v w : Fin 3 → R
    ⊢ Eq (Matrix.vecCons (HSub.hSub (HMul.hMul (u 1) (Matrix.vecCons (HSub.hSub (H …
  -/
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
  apply vec3_eq <;> norm_num <;> ring
                                 /-
                                   🎉 no goals
                                 -/


/-- The three-dimensional vectors together with the operations + and ×₃ form a Lie ring.
    Note we do not make this an instance as a conflicting one already exists
    via `LieRing.ofAssociativeRing`. -/
def Cross.lieRing : LieRing (Fin 3 → R) :=
  { Pi.addCommGroup with
    bracket := fun u v => u ×₃ v
    add_lie := LinearMap.map_add₂ _
    lie_add := fun _ => LinearMap.map_add _
    lie_self := cross_self
    leibniz_lie := leibniz_cross }


theorem cross_cross (u v w : Fin 3 → R) : u ×₃ v ×₃ w = u ×₃ (v ×₃ w) - v ×₃ (u ×₃ w) :=
  lie_lie u v w


/-- **Jacobi identity**: For a cross product of three vectors,
    their sum over the three even permutations is equal to the zero vector. -/
theorem jacobi_cross (u v w : Fin 3 → R) : u ×₃ (v ×₃ w) + v ×₃ (w ×₃ u) + w ×₃ (u ×₃ v) = 0 :=
  lie_jacobi u v w


lemma crossProduct_ne_zero_iff_linearIndependent {F : Type*} [Field F] {v w : Fin 3 → F} :
    crossProduct v w ≠ 0 ↔ LinearIndependent F ![v, w] := by
  /-
    F : Type u_2
    inst✝ : Field F
    v w : Fin 3 → F
    ⊢ Iff (Ne ((crossProduct v) w) 0) (LinearIndependent F (Matrix.vecCons v (Matr …
  -/
  rw [not_iff_comm]
  /-
    F : Type u_2
    inst✝ : Field F
    v w : Fin 3 → F
    ⊢ Iff (Not (LinearIndependent F (Matrix.vecCons v (Matrix.vecCons w Matrix.vec …
  -/
  by_cases hv : v = 0
    /-
      case pos
      F : Type u_2
      inst✝ : Field F
      v w : Fin 3 → F
      hv : Eq v 0
      ⊢ Iff (Not (LinearIndependent F (Matrix.vecCons v (Matrix.vecCons w Matrix.vec …
    -/
  · rw [hv, map_zero, LinearMap.zero_apply, eq_self, iff_true]
    /-
      case pos
      F : Type u_2
      inst✝ : Field F
      v w : Fin 3 → F
      hv : Eq v 0
      ⊢ Not (LinearIndependent F (Matrix.vecCons 0 (Matrix.vecCons w Matrix.vecEmpty …
    -/
    exact fun h ↦ h.ne_zero 0 rfl
    /-
      🎉 no goals
    -/
  /-
    case neg
    F : Type u_2
    inst✝ : Field F
    v w : Fin 3 → F
    hv : Not (Eq v 0)
    ⊢ Iff (Not (LinearIndependent F (Matrix.vecCons v (Matrix.vecCons w Matrix.vec …
  -/
  constructor
    /-
      case neg.mp
      F : Type u_2
      inst✝ : Field F
      v w : Fin 3 → F
      hv : Not (Eq v 0)
      ⊢ Not (LinearIndependent F (Matrix.vecCons v (Matrix.vecCons w Matrix.vecEmpty …
    -/
  · rw [LinearIndependent.pair_iff' hv, not_forall_not]
    /-
      case neg.mp
      F : Type u_2
      inst✝ : Field F
      v w : Fin 3 → F
      hv : Not (Eq v 0)
      ⊢ (Exists fun x => Eq (HSMul.hSMul x v) w) → Eq ((crossProduct v) w) 0
    -/
    rintro ⟨a, rfl⟩
    /-
      case neg.mp.intro
      F : Type u_2
      inst✝ : Field F
      v : Fin 3 → F
      hv : Not (Eq v 0)
      a : F
      ⊢ Eq ((crossProduct v) (HSMul.hSMul a v)) 0
    -/
    rw [LinearMap.map_smul, cross_self, smul_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg.mpr
    F : Type u_2
    inst✝ : Field F
    v w : Fin 3 → F
    hv : Not (Eq v 0)
    ⊢ Eq ((crossProduct v) w) 0 → Not (LinearIndependent F (Matrix.vecCons v (Matr …
  -/
  have hv' : v = ![v 0, v 1, v 2] := by simp [← List.ofFn_inj]
  /-
    case neg.mpr
    F : Type u_2
    inst✝ : Field F
    v w : Fin 3 → F
    hv : Not (Eq v 0)
    hv' : Eq v (Matrix.vecCons (v 0) (Matrix.vecCons (v 1) (Matrix.vecCons (v 2) M …
    ⊢ Eq ((crossProduct v) w) 0 → Not (LinearIndependent F (Matrix.vecCons v (Matr …
  -/
  have hw' : w = ![w 0, w 1, w 2] := by simp [← List.ofFn_inj]
  /-
    case neg.mpr
    F : Type u_2
    inst✝ : Field F
    v w : Fin 3 → F
    hv : Not (Eq v 0)
    hv' : Eq v (Matrix.vecCons (v 0) (Matrix.vecCons (v 1) (Matrix.vecCons (v 2) M …
    hw' : Eq w (Matrix.vecCons (w 0) (Matrix.vecCons (w 1) (Matrix.vecCons (w 2) M …
    ⊢ Eq ((crossProduct v) w) 0 → Not (LinearIndependent F (Matrix.vecCons v (Matr …
  -/
  intro h1 h2
  /-
    case neg.mpr
    F : Type u_2
    inst✝ : Field F
    v w : Fin 3 → F
    hv : Not (Eq v 0)
    hv' : Eq v (Matrix.vecCons (v 0) (Matrix.vecCons (v 1) (Matrix.vecCons (v 2) M …
    hw' : Eq w (Matrix.vecCons (w 0) (Matrix.vecCons (w 1) (Matrix.vecCons (w 2) M …
    h1 : Eq ((crossProduct v) w) 0
    h2 : LinearIndependent F (Matrix.vecCons v (Matrix.vecCons w Matrix.vecEmpty))
    ⊢ False
  -/
  simp_rw [cross_apply, cons_eq_zero_iff, zero_empty, and_true, sub_eq_zero] at h1
  /-
    case neg.mpr
    F : Type u_2
    inst✝ : Field F
    v w : Fin 3 → F
    hv : Not (Eq v 0)
    hv' : Eq v (Matrix.vecCons (v 0) (Matrix.vecCons (v 1) (Matrix.vecCons (v 2) M …
    hw' : Eq w (Matrix.vecCons (w 0) (Matrix.vecCons (w 1) (Matrix.vecCons (w 2) M …
    h2 : LinearIndependent F (Matrix.vecCons v (Matrix.vecCons w Matrix.vecEmpty))
    h1 : And (Eq (HMul.hMul (v 1) (w 2)) (HMul.hMul (v 2) (w 1))) (And (Eq (HMul.h …
    ⊢ False
  -/
  have h20 := LinearIndependent.pair_iff.mp h2 (- w 0) (v 0)
  /-
    case neg.mpr
    F : Type u_2
    inst✝ : Field F
    v w : Fin 3 → F
    hv : Not (Eq v 0)
    hv' : Eq v (Matrix.vecCons (v 0) (Matrix.vecCons (v 1) (Matrix.vecCons (v 2) M …
    hw' : Eq w (Matrix.vecCons (w 0) (Matrix.vecCons (w 1) (Matrix.vecCons (w 2) M …
    h2 : LinearIndependent F (Matrix.vecCons v (Matrix.vecCons w Matrix.vecEmpty))
    h1 : And (Eq (HMul.hMul (v 1) (w 2)) (HMul.hMul (v 2) (w 1))) (And (Eq (HMul.h …
    h20 : Eq (HAdd.hAdd (HSMul.hSMul (Neg.neg (w 0)) v) (HSMul.hSMul (v 0) w)) 0 → …
    ⊢ False
  -/
  have h21 := LinearIndependent.pair_iff.mp h2 (- w 1) (v 1)
  /-
    case neg.mpr
    F : Type u_2
    inst✝ : Field F
    v w : Fin 3 → F
    hv : Not (Eq v 0)
    hv' : Eq v (Matrix.vecCons (v 0) (Matrix.vecCons (v 1) (Matrix.vecCons (v 2) M …
    hw' : Eq w (Matrix.vecCons (w 0) (Matrix.vecCons (w 1) (Matrix.vecCons (w 2) M …
    h2 : LinearIndependent F (Matrix.vecCons v (Matrix.vecCons w Matrix.vecEmpty))
    h1 : And (Eq (HMul.hMul (v 1) (w 2)) (HMul.hMul (v 2) (w 1))) (And (Eq (HMul.h …
    h20 : Eq (HAdd.hAdd (HSMul.hSMul (Neg.neg (w 0)) v) (HSMul.hSMul (v 0) w)) 0 → …
    h21 : Eq (HAdd.hAdd (HSMul.hSMul (Neg.neg (w 1)) v) (HSMul.hSMul (v 1) w)) 0 → …
    ⊢ False
  -/
  have h22 := LinearIndependent.pair_iff.mp h2 (- w 2) (v 2)
  /-
    case neg.mpr
    F : Type u_2
    inst✝ : Field F
    v w : Fin 3 → F
    hv : Not (Eq v 0)
    hv' : Eq v (Matrix.vecCons (v 0) (Matrix.vecCons (v 1) (Matrix.vecCons (v 2) M …
    hw' : Eq w (Matrix.vecCons (w 0) (Matrix.vecCons (w 1) (Matrix.vecCons (w 2) M …
    h2 : LinearIndependent F (Matrix.vecCons v (Matrix.vecCons w Matrix.vecEmpty))
    h1 : And (Eq (HMul.hMul (v 1) (w 2)) (HMul.hMul (v 2) (w 1))) (And (Eq (HMul.h …
    h20 : Eq (HAdd.hAdd (HSMul.hSMul (Neg.neg (w 0)) v) (HSMul.hSMul (v 0) w)) 0 → …
    h21 : Eq (HAdd.hAdd (HSMul.hSMul (Neg.neg (w 1)) v) (HSMul.hSMul (v 1) w)) 0 → …
    h22 : Eq (HAdd.hAdd (HSMul.hSMul (Neg.neg (w 2)) v) (HSMul.hSMul (v 2) w)) 0 → …
    ⊢ False
  -/
  rw [neg_smul, neg_add_eq_zero, hv', hw', smul_vec3, smul_vec3, ← hv', ← hw'] at h20 h21 h22
  /-
    case neg.mpr
    F : Type u_2
    inst✝ : Field F
    v w : Fin 3 → F
    hv : Not (Eq v 0)
    hv' : Eq v (Matrix.vecCons (v 0) (Matrix.vecCons (v 1) (Matrix.vecCons (v 2) M …
    hw' : Eq w (Matrix.vecCons (w 0) (Matrix.vecCons (w 1) (Matrix.vecCons (w 2) M …
    h2 : LinearIndependent F (Matrix.vecCons v (Matrix.vecCons w Matrix.vecEmpty))
    h1 : And (Eq (HMul.hMul (v 1) (w 2)) (HMul.hMul (v 2) (w 1))) (And (Eq (HMul.h …
    h20 : Eq (Matrix.vecCons (HSMul.hSMul (w 0) (v 0)) (Matrix.vecCons (HSMul.hSMu …
    h21 : Eq (Matrix.vecCons (HSMul.hSMul (w 1) (v 0)) (Matrix.vecCons (HSMul.hSMu …
    h22 : Eq (Matrix.vecCons (HSMul.hSMul (w 2) (v 0)) (Matrix.vecCons (HSMul.hSMu …
    ⊢ False
  -/
  simp only [smul_eq_mul, mul_comm (w 0), mul_comm (w 1), mul_comm (w 2), h1] at h20 h21 h22
  /-
    case neg.mpr
    F : Type u_2
    inst✝ : Field F
    v w : Fin 3 → F
    hv : Not (Eq v 0)
    hv' : Eq v (Matrix.vecCons (v 0) (Matrix.vecCons (v 1) (Matrix.vecCons (v 2) M …
    hw' : Eq w (Matrix.vecCons (w 0) (Matrix.vecCons (w 1) (Matrix.vecCons (w 2) M …
    h2 : LinearIndependent F (Matrix.vecCons v (Matrix.vecCons w Matrix.vecEmpty))
    h1 : And (Eq (HMul.hMul (v 1) (w 2)) (HMul.hMul (v 2) (w 1))) (And (Eq (HMul.h …
    h20 : True → And (Eq (Neg.neg (w 0)) 0) (Eq (v 0) 0)
    h21 : True → And (Eq (Neg.neg (w 1)) 0) (Eq (v 1) 0)
    h22 : True → And (Eq (Neg.neg (w 2)) 0) (Eq (v 2) 0)
    ⊢ False
  -/
  rw [hv', cons_eq_zero_iff, cons_eq_zero_iff, cons_eq_zero_iff, zero_empty] at hv
  /-
    case neg.mpr
    F : Type u_2
    inst✝ : Field F
    v w : Fin 3 → F
    hv : Not (And (Eq (v 0) 0) (And (Eq (v 1) 0) (And (Eq (v 2) 0) (Eq Matrix.vecE …
    hv' : Eq v (Matrix.vecCons (v 0) (Matrix.vecCons (v 1) (Matrix.vecCons (v 2) M …
    hw' : Eq w (Matrix.vecCons (w 0) (Matrix.vecCons (w 1) (Matrix.vecCons (w 2) M …
    h2 : LinearIndependent F (Matrix.vecCons v (Matrix.vecCons w Matrix.vecEmpty))
    h1 : And (Eq (HMul.hMul (v 1) (w 2)) (HMul.hMul (v 2) (w 1))) (And (Eq (HMul.h …
    h20 : True → And (Eq (Neg.neg (w 0)) 0) (Eq (v 0) 0)
    h21 : True → And (Eq (Neg.neg (w 1)) 0) (Eq (v 1) 0)
    h22 : True → And (Eq (Neg.neg (w 2)) 0) (Eq (v 2) 0)
    ⊢ False
  -/
  exact hv ⟨(h20 trivial).2, (h21 trivial).2, (h22 trivial).2, rfl⟩
  /-
    🎉 no goals
  -/

