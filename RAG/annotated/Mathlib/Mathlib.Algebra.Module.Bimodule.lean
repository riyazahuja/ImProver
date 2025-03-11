/-- A constructor for a subbimodule which demands closure under the two sets of scalars
individually, rather than jointly via their tensor product.

Note that `R` plays no role but it is convenient to make this generalisation to support the cases
`R = ℕ` and `R = ℤ` which both show up naturally. See also `Subbimodule.baseChange`. -/
@[simps]
def mk (p : AddSubmonoid M) (hA : ∀ (a : A) {m : M}, m ∈ p → a • m ∈ p)
    (hB : ∀ (b : B) {m : M}, m ∈ p → b • m ∈ p) : Submodule (A ⊗[R] B) M :=
  { p with
    carrier := p
    smul_mem' := fun ab m =>
                                                 /-
                                                   R : Type u_1
                                                   A : Type u_2
                                                   B : Type u_3
                                                   M : Type u_4
                                                   inst✝¹¹ : CommSemiring R
                                                   inst✝¹⁰ : AddCommMonoid M
                                                   inst✝⁹ : Module R M
                                                   inst✝⁸ : Semiring A
                                                   inst✝⁷ : Semiring B
                                                   inst✝⁶ : Module A M
                                                   inst✝⁵ : Module B M
                                                   inst✝⁴ : Algebra R A
                                                   inst✝³ : Algebra R B
                                                   inst✝² : IsScalarTower R A M
                                                   inst✝¹ : IsScalarTower R B M
                                                   inst✝ : SMulCommClass A B M
                                                   p : AddSubmonoid M
                                                   hA : ∀ (a : A) {m : M}, Membership.mem p m → Membership.mem p (HSMul.hSMul a m)
                                                   hB : ∀ (b : B) {m : M}, Membership.mem p m → Membership.mem p (HSMul.hSMul b m)
                                                   ab : TensorProduct R A B
                                                   m : M
                                                   x✝ : Membership.mem { carrier := ↑p, add_mem' := ⋯, zero_mem' := ⋯ }.carrier m
                                                   ⊢ Membership.mem { carrier := ↑p, add_mem' := ⋯, zero_mem' := ⋯ }.carrier (HSM …
                                                 -/
      TensorProduct.induction_on ab (fun _ => by simpa only [zero_smul] using p.zero_mem)
                                                 /-
                                                   🎉 no goals
                                                 -/
                          /-
                            R : Type u_1
                            A : Type u_2
                            B : Type u_3
                            M : Type u_4
                            inst✝¹¹ : CommSemiring R
                            inst✝¹⁰ : AddCommMonoid M
                            inst✝⁹ : Module R M
                            inst✝⁸ : Semiring A
                            inst✝⁷ : Semiring B
                            inst✝⁶ : Module A M
                            inst✝⁵ : Module B M
                            inst✝⁴ : Algebra R A
                            inst✝³ : Algebra R B
                            inst✝² : IsScalarTower R A M
                            inst✝¹ : IsScalarTower R B M
                            inst✝ : SMulCommClass A B M
                            p : AddSubmonoid M
                            hA : ∀ (a : A) {m : M}, Membership.mem p m → Membership.mem p (HSMul.hSMul a m)
                            hB : ∀ (b : B) {m : M}, Membership.mem p m → Membership.mem p (HSMul.hSMul b m)
                            ab : TensorProduct R A B
                            m : M
                            a : A
                            b : B
                            hm : Membership.mem { carrier := ↑p, add_mem' := ⋯, zero_mem' := ⋯ }.carrier m
                            ⊢ Membership.mem { carrier := ↑p, add_mem' := ⋯, zero_mem' := ⋯ }.carrier (HSM …
                          -/
        (fun a b hm => by simpa only [TensorProduct.Algebra.smul_def] using hA a (hB b hm))
                          /-
                            🎉 no goals
                          -/
                               /-
                                 R : Type u_1
                                 A : Type u_2
                                 B : Type u_3
                                 M : Type u_4
                                 inst✝¹¹ : CommSemiring R
                                 inst✝¹⁰ : AddCommMonoid M
                                 inst✝⁹ : Module R M
                                 inst✝⁸ : Semiring A
                                 inst✝⁷ : Semiring B
                                 inst✝⁶ : Module A M
                                 inst✝⁵ : Module B M
                                 inst✝⁴ : Algebra R A
                                 inst✝³ : Algebra R B
                                 inst✝² : IsScalarTower R A M
                                 inst✝¹ : IsScalarTower R B M
                                 inst✝ : SMulCommClass A B M
                                 p : AddSubmonoid M
                                 hA : ∀ (a : A) {m : M}, Membership.mem p m → Membership.mem p (HSMul.hSMul a m)
                                 hB : ∀ (b : B) {m : M}, Membership.mem p m → Membership.mem p (HSMul.hSMul b m)
                                 ab : TensorProduct R A B
                                 m : M
                                 z w : TensorProduct R A B
                                 hz : Membership.mem { carrier := ↑p, add_mem' := ⋯, zero_mem' := ⋯ }.carrier m …
                                 hw : Membership.mem { carrier := ↑p, add_mem' := ⋯, zero_mem' := ⋯ }.carrier m …
                                 hm : Membership.mem { carrier := ↑p, add_mem' := ⋯, zero_mem' := ⋯ }.carrier m
                                 ⊢ Membership.mem { carrier := ↑p, add_mem' := ⋯, zero_mem' := ⋯ }.carrier (HSM …
                               -/
        fun z w hz hw hm => by simpa only [add_smul] using p.add_mem (hz hm) (hw hm) }
                               /-
                                 🎉 no goals
                               -/


theorem smul_mem (p : Submodule (A ⊗[R] B) M) (a : A) {m : M} (hm : m ∈ p) : a • m ∈ p := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    M : Type u_4
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : AddCommMonoid M
    inst✝⁹ : Module R M
    inst✝⁸ : Semiring A
    inst✝⁷ : Semiring B
    inst✝⁶ : Module A M
    inst✝⁵ : Module B M
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    inst✝² : IsScalarTower R A M
    inst✝¹ : IsScalarTower R B M
    inst✝ : SMulCommClass A B M
    p : Submodule (TensorProduct R A B) M
    a : A
    m : M
    hm : Membership.mem p m
    ⊢ Membership.mem p (HSMul.hSMul a m)
  -/
  suffices a • m = a ⊗ₜ[R] (1 : B) • m by exact this.symm ▸ p.smul_mem _ hm
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    M : Type u_4
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : AddCommMonoid M
    inst✝⁹ : Module R M
    inst✝⁸ : Semiring A
    inst✝⁷ : Semiring B
    inst✝⁶ : Module A M
    inst✝⁵ : Module B M
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    inst✝² : IsScalarTower R A M
    inst✝¹ : IsScalarTower R B M
    inst✝ : SMulCommClass A B M
    p : Submodule (TensorProduct R A B) M
    a : A
    m : M
    hm : Membership.mem p m
    ⊢ Eq (HSMul.hSMul a m) (HSMul.hSMul (TensorProduct.tmul R a 1) m)
  -/
  simp [TensorProduct.Algebra.smul_def]
  /-
    🎉 no goals
  -/


theorem smul_mem' (p : Submodule (A ⊗[R] B) M) (b : B) {m : M} (hm : m ∈ p) : b • m ∈ p := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    M : Type u_4
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : AddCommMonoid M
    inst✝⁹ : Module R M
    inst✝⁸ : Semiring A
    inst✝⁷ : Semiring B
    inst✝⁶ : Module A M
    inst✝⁵ : Module B M
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    inst✝² : IsScalarTower R A M
    inst✝¹ : IsScalarTower R B M
    inst✝ : SMulCommClass A B M
    p : Submodule (TensorProduct R A B) M
    b : B
    m : M
    hm : Membership.mem p m
    ⊢ Membership.mem p (HSMul.hSMul b m)
  -/
  suffices b • m = (1 : A) ⊗ₜ[R] b • m by exact this.symm ▸ p.smul_mem _ hm
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    M : Type u_4
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : AddCommMonoid M
    inst✝⁹ : Module R M
    inst✝⁸ : Semiring A
    inst✝⁷ : Semiring B
    inst✝⁶ : Module A M
    inst✝⁵ : Module B M
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    inst✝² : IsScalarTower R A M
    inst✝¹ : IsScalarTower R B M
    inst✝ : SMulCommClass A B M
    p : Submodule (TensorProduct R A B) M
    b : B
    m : M
    hm : Membership.mem p m
    ⊢ Eq (HSMul.hSMul b m) (HSMul.hSMul (TensorProduct.tmul R 1 b) m)
  -/
  simp [TensorProduct.Algebra.smul_def]
  /-
    🎉 no goals
  -/


/-- If `A` and `B` are also `Algebra`s over yet another set of scalars `S` then we may "base change"
from `R` to `S`. -/
@[simps!]
def baseChange (S : Type*) [CommSemiring S] [Module S M] [Algebra S A] [Algebra S B]
    [IsScalarTower S A M] [IsScalarTower S B M] (p : Submodule (A ⊗[R] B) M) :
    Submodule (A ⊗[S] B) M :=
  mk p.toAddSubmonoid (smul_mem p) (smul_mem' p)


/-- Forgetting the `B` action, a `Submodule` over `A ⊗[R] B` is just a `Submodule` over `A`. -/
@[simps]
def toSubmodule (p : Submodule (A ⊗[R] B) M) : Submodule A M :=
  { p with
    carrier := p
    smul_mem' := smul_mem p }


/-- Forgetting the `A` action, a `Submodule` over `A ⊗[R] B` is just a `Submodule` over `B`. -/
@[simps]
def toSubmodule' (p : Submodule (A ⊗[R] B) M) : Submodule B M :=
  { p with
    carrier := p
    smul_mem' := smul_mem' p }


/-- A `Submodule` over `R ⊗[ℕ] S` is naturally also a `Submodule` over the canonically-isomorphic
ring `R ⊗[ℤ] S`. -/
@[simps!]
def toSubbimoduleInt (p : Submodule (R ⊗[ℕ] S) M) : Submodule (R ⊗[ℤ] S) M :=
  baseChange ℤ p


/-- A `Submodule` over `R ⊗[ℤ] S` is naturally also a `Submodule` over the canonically-isomorphic
ring `R ⊗[ℕ] S`. -/
@[simps!]
def toSubbimoduleNat (p : Submodule (R ⊗[ℤ] S) M) : Submodule (R ⊗[ℕ] S) M :=
  baseChange ℕ p


