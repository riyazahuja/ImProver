theorem mul_toSubmodule_le (S T : Subalgebra R A) :
    (Subalgebra.toSubmodule S)* (Subalgebra.toSubmodule T) ≤ Subalgebra.toSubmodule (S ⊔ T) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    S T : Subalgebra R A
    ⊢ LE.le (HMul.hMul (Subalgebra.toSubmodule S) (Subalgebra.toSubmodule T)) (Sub …
  -/
  rw [Submodule.mul_le]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    S T : Subalgebra R A
    ⊢ ∀ (m : A), Membership.mem (Subalgebra.toSubmodule S) m → ∀ (n : A), Membersh …
  -/
  intro y hy z hz
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    S T : Subalgebra R A
    y : A
    hy : Membership.mem (Subalgebra.toSubmodule S) y
    z : A
    hz : Membership.mem (Subalgebra.toSubmodule T) z
    ⊢ Membership.mem (Subalgebra.toSubmodule (Max.max S T)) (HMul.hMul y z)
  -/
  show y * z ∈ S ⊔ T
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    S T : Subalgebra R A
    y : A
    hy : Membership.mem (Subalgebra.toSubmodule S) y
    z : A
    hz : Membership.mem (Subalgebra.toSubmodule T) z
    ⊢ Membership.mem (Max.max S T) (HMul.hMul y z)
  -/
  exact mul_mem (Algebra.mem_sup_left hy) (Algebra.mem_sup_right hz)
  /-
    🎉 no goals
  -/


/-- As submodules, subalgebras are idempotent. -/
@[simp]
theorem mul_self (S : Subalgebra R A) : (Subalgebra.toSubmodule S) * (Subalgebra.toSubmodule S)
    = (Subalgebra.toSubmodule S) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    S : Subalgebra R A
    ⊢ Eq (HMul.hMul (Subalgebra.toSubmodule S) (Subalgebra.toSubmodule S)) (Subalg …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      S : Subalgebra R A
      ⊢ LE.le (HMul.hMul (Subalgebra.toSubmodule S) (Subalgebra.toSubmodule S)) (Sub …
    -/
  · refine (mul_toSubmodule_le _ _).trans_eq ?_
    /-
      case a
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      S : Subalgebra R A
      ⊢ Eq (Subalgebra.toSubmodule (Max.max S S)) (Subalgebra.toSubmodule S)
    -/
    rw [sup_idem]
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      S : Subalgebra R A
      ⊢ LE.le (Subalgebra.toSubmodule S) (HMul.hMul (Subalgebra.toSubmodule S) (Suba …
    -/
  · intro x hx1
    /-
      case a
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      S : Subalgebra R A
      x : A
      hx1 : Membership.mem (Subalgebra.toSubmodule S) x
      ⊢ Membership.mem (HMul.hMul (Subalgebra.toSubmodule S) (Subalgebra.toSubmodule …
    -/
    rw [← mul_one x]
    /-
      case a
      R : Type u_1
      A : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      S : Subalgebra R A
      x : A
      hx1 : Membership.mem (Subalgebra.toSubmodule S) x
      ⊢ Membership.mem (HMul.hMul (Subalgebra.toSubmodule S) (Subalgebra.toSubmodule …
    -/
    exact Submodule.mul_mem_mul hx1 (show (1 : A) ∈ S from one_mem S)
    /-
      🎉 no goals
    -/


/-- When `A` is commutative, `Subalgebra.mul_toSubmodule_le` is strict. -/
theorem mul_toSubmodule {R : Type*} {A : Type*} [CommSemiring R] [CommSemiring A] [Algebra R A]
    (S T : Subalgebra R A) : (Subalgebra.toSubmodule S) * (Subalgebra.toSubmodule T)
        = Subalgebra.toSubmodule (S ⊔ T) := by
  /-
    R : Type u_3
    A : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    S T : Subalgebra R A
    ⊢ Eq (HMul.hMul (Subalgebra.toSubmodule S) (Subalgebra.toSubmodule T)) (Subalg …
  -/
  refine le_antisymm (mul_toSubmodule_le _ _) ?_
  /-
    R : Type u_3
    A : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    S T : Subalgebra R A
    ⊢ LE.le (Subalgebra.toSubmodule (Max.max S T)) (HMul.hMul (Subalgebra.toSubmod …
  -/
  rintro x (hx : x ∈ Algebra.adjoin R (S ∪ T : Set A))
  refine
    Algebra.adjoin_induction (fun x hx => ?_) (fun r => ?_) (fun _ _ _ _ => Submodule.add_mem _)
      (fun x y _ _ hx hy => ?_) hx
    /-
      case refine_1
      R : Type u_3
      A : Type u_4
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      S T : Subalgebra R A
      x✝ : A
      hx✝ : Membership.mem (Algebra.adjoin R (Union.union ↑S ↑T)) x✝
      x : A
      hx : Membership.mem (Union.union ↑S ↑T) x
      ⊢ Membership.mem (HMul.hMul (Subalgebra.toSubmodule S) (Subalgebra.toSubmodule …
    -/
  · rcases hx with hxS | hxT
      /-
        case refine_1.inl
        R : Type u_3
        A : Type u_4
        inst✝² : CommSemiring R
        inst✝¹ : CommSemiring A
        inst✝ : Algebra R A
        S T : Subalgebra R A
        x✝ : A
        hx : Membership.mem (Algebra.adjoin R (Union.union ↑S ↑T)) x✝
        x : A
        hxS : Membership.mem (↑S) x
        ⊢ Membership.mem (HMul.hMul (Subalgebra.toSubmodule S) (Subalgebra.toSubmodule …
      -/
    · rw [← mul_one x]
      /-
        case refine_1.inl
        R : Type u_3
        A : Type u_4
        inst✝² : CommSemiring R
        inst✝¹ : CommSemiring A
        inst✝ : Algebra R A
        S T : Subalgebra R A
        x✝ : A
        hx : Membership.mem (Algebra.adjoin R (Union.union ↑S ↑T)) x✝
        x : A
        hxS : Membership.mem (↑S) x
        ⊢ Membership.mem (HMul.hMul (Subalgebra.toSubmodule S) (Subalgebra.toSubmodule …
      -/
      exact Submodule.mul_mem_mul hxS (show (1 : A) ∈ T from one_mem T)
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr
        R : Type u_3
        A : Type u_4
        inst✝² : CommSemiring R
        inst✝¹ : CommSemiring A
        inst✝ : Algebra R A
        S T : Subalgebra R A
        x✝ : A
        hx : Membership.mem (Algebra.adjoin R (Union.union ↑S ↑T)) x✝
        x : A
        hxT : Membership.mem (↑T) x
        ⊢ Membership.mem (HMul.hMul (Subalgebra.toSubmodule S) (Subalgebra.toSubmodule …
      -/
    · rw [← one_mul x]
      /-
        case refine_1.inr
        R : Type u_3
        A : Type u_4
        inst✝² : CommSemiring R
        inst✝¹ : CommSemiring A
        inst✝ : Algebra R A
        S T : Subalgebra R A
        x✝ : A
        hx : Membership.mem (Algebra.adjoin R (Union.union ↑S ↑T)) x✝
        x : A
        hxT : Membership.mem (↑T) x
        ⊢ Membership.mem (HMul.hMul (Subalgebra.toSubmodule S) (Subalgebra.toSubmodule …
      -/
      exact Submodule.mul_mem_mul (show (1 : A) ∈ S from one_mem S) hxT
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      R : Type u_3
      A : Type u_4
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      S T : Subalgebra R A
      x : A
      hx : Membership.mem (Algebra.adjoin R (Union.union ↑S ↑T)) x
      r : R
      ⊢ Membership.mem (HMul.hMul (Subalgebra.toSubmodule S) (Subalgebra.toSubmodule …
    -/
  · rw [← one_mul (algebraMap _ _ _)]
    /-
      case refine_2
      R : Type u_3
      A : Type u_4
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      S T : Subalgebra R A
      x : A
      hx : Membership.mem (Algebra.adjoin R (Union.union ↑S ↑T)) x
      r : R
      ⊢ Membership.mem (HMul.hMul (Subalgebra.toSubmodule S) (Subalgebra.toSubmodule …
    -/
    exact Submodule.mul_mem_mul (show (1 : A) ∈ S from one_mem S) (algebraMap_mem T _)
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    R : Type u_3
    A : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    S T : Subalgebra R A
    x✝² : A
    hx✝ : Membership.mem (Algebra.adjoin R (Union.union ↑S ↑T)) x✝²
    x y : A
    x✝¹ : Membership.mem (Algebra.adjoin R (Union.union ↑S ↑T)) x
    x✝ : Membership.mem (Algebra.adjoin R (Union.union ↑S ↑T)) y
    hx : Membership.mem (HMul.hMul (Subalgebra.toSubmodule S) (Subalgebra.toSubmod …
    hy : Membership.mem (HMul.hMul (Subalgebra.toSubmodule S) (Subalgebra.toSubmod …
    ⊢ Membership.mem (HMul.hMul (Subalgebra.toSubmodule S) (Subalgebra.toSubmodule …
  -/
  have := Submodule.mul_mem_mul hx hy
  rwa [mul_assoc, mul_comm _ (Subalgebra.toSubmodule T), ← mul_assoc _ _ (Subalgebra.toSubmodule S),
    mul_self, mul_comm (Subalgebra.toSubmodule T), ← mul_assoc, mul_self] at this


/-- The action on a subalgebra corresponding to applying the action to every element.

This is available as an instance in the `Pointwise` locale. -/
protected def pointwiseMulAction : MulAction R' (Subalgebra R A) where
  smul a S := S.map (MulSemiringAction.toAlgHom _ _ a)
  one_smul S := (congr_arg (fun f => S.map f) (AlgHom.ext <| one_smul R')).trans S.map_id
  mul_smul _a₁ _a₂ S :=
    (congr_arg (fun f => S.map f) (AlgHom.ext <| mul_smul _ _)).trans (S.map_map _ _).symm


@[simp]
theorem coe_pointwise_smul (m : R') (S : Subalgebra R A) : ↑(m • S) = m • (S : Set A) :=
  rfl


@[simp]
theorem pointwise_smul_toSubsemiring (m : R') (S : Subalgebra R A) :
    (m • S).toSubsemiring = m • S.toSubsemiring :=
  rfl


@[simp]
theorem pointwise_smul_toSubmodule (m : R') (S : Subalgebra R A) :
    Subalgebra.toSubmodule (m • S) = m • Subalgebra.toSubmodule S :=
  rfl


@[simp]
theorem pointwise_smul_toSubring {R' R A : Type*} [Semiring R'] [CommRing R] [Ring A]
    [MulSemiringAction R' A] [Algebra R A] [SMulCommClass R' R A] (m : R') (S : Subalgebra R A) :
    (m • S).toSubring = m • S.toSubring :=
  rfl


theorem smul_mem_pointwise_smul (m : R') (r : A) (S : Subalgebra R A) : r ∈ S → m • r ∈ m • S :=
  (Set.smul_mem_smul_set : _ → _ ∈ m • (S : Set A))


instance : CovariantClass R' (Subalgebra R A) HSMul.hSMul LE.le :=
  ⟨fun _ _ => map_mono⟩


