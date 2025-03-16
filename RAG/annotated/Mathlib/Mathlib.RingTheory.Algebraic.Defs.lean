/-- An element of an R-algebra is algebraic over R if it is a root of a nonzero polynomial
with coefficients in R. -/
@[stacks 09GC "Algebraic elements"]
def IsAlgebraic (x : A) : Prop :=
  ∃ p : R[X], p ≠ 0 ∧ aeval x p = 0


/-- An element of an R-algebra is transcendental over R if it is not algebraic over R. -/
def Transcendental (x : A) : Prop :=
  ¬IsAlgebraic R x


/-- An element `x` is transcendental over `R` if and only if for any polynomial `p`,
`Polynomial.aeval x p = 0` implies `p = 0`. This is similar to `algebraicIndependent_iff`. -/
theorem transcendental_iff {x : A} :
    Transcendental R x ↔ ∀ p : R[X], aeval x p = 0 → p = 0 := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    x : A
    ⊢ Iff (Transcendental R x) (∀ (p : Polynomial R), Eq ((Polynomial.aeval x) p)  …
  -/
  rw [Transcendental, IsAlgebraic, not_exists]
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    x : A
    ⊢ Iff (∀ (x_1 : Polynomial R), Not (And (Ne x_1 0) (Eq ((Polynomial.aeval x) x …
  -/
  congr! 1; tauto
            /-
              🎉 no goals
            -/


/-- A subalgebra is algebraic if all its elements are algebraic. -/
nonrec
def Subalgebra.IsAlgebraic (S : Subalgebra R A) : Prop :=
  ∀ x ∈ S, IsAlgebraic R x


/-- An algebra is algebraic if all its elements are algebraic. -/
@[stacks 09GC "Algebraic extensions"]
protected class Algebra.IsAlgebraic : Prop where
  isAlgebraic : ∀ x : A, IsAlgebraic R x


/-- An algebra is transcendental if some element is transcendental. -/
protected class Algebra.Transcendental : Prop where
  transcendental : ∃ x : A, Transcendental R x


lemma Algebra.isAlgebraic_def : Algebra.IsAlgebraic R A ↔ ∀ x : A, IsAlgebraic R x :=
  ⟨fun ⟨h⟩ ↦ h, fun h ↦ ⟨h⟩⟩


lemma Algebra.transcendental_def : Algebra.Transcendental R A ↔ ∃ x : A, Transcendental R x :=
  ⟨fun ⟨h⟩ ↦ h, fun h ↦ ⟨h⟩⟩


theorem Algebra.transcendental_iff_not_isAlgebraic :
    Algebra.Transcendental R A ↔ ¬ Algebra.IsAlgebraic R A := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    ⊢ Iff (Algebra.Transcendental R A) (Not (Algebra.IsAlgebraic R A))
  -/
  simp [isAlgebraic_def, transcendental_def, Transcendental]
  /-
    🎉 no goals
  -/


/-- A subalgebra is algebraic if and only if it is algebraic as an algebra. -/
theorem Subalgebra.isAlgebraic_iff (S : Subalgebra R A) :
    S.IsAlgebraic ↔ Algebra.IsAlgebraic R S := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    S : Subalgebra R A
    ⊢ Iff S.IsAlgebraic (Algebra.IsAlgebraic R (Subtype fun x => Membership.mem S  …
  -/
  delta Subalgebra.IsAlgebraic
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    S : Subalgebra R A
    ⊢ Iff (∀ (x : A), Membership.mem S x → _root_.IsAlgebraic R x) (Algebra.IsAlge …
  -/
  rw [Subtype.forall', Algebra.isAlgebraic_def]
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    S : Subalgebra R A
    ⊢ Iff (∀ (x : Subtype fun a => Membership.mem S a), _root_.IsAlgebraic R ↑x) ( …
  -/
  refine forall_congr' fun x => exists_congr fun p => and_congr Iff.rfl ?_
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    S : Subalgebra R A
    x : Subtype fun a => Membership.mem S a
    p : Polynomial R
    ⊢ Iff (Eq ((Polynomial.aeval ↑x) p) 0) (Eq ((Polynomial.aeval x) p) 0)
  -/
  have h : Function.Injective S.val := Subtype.val_injective
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    S : Subalgebra R A
    x : Subtype fun a => Membership.mem S a
    p : Polynomial R
    h : Function.Injective ⇑S.val
    ⊢ Iff (Eq ((Polynomial.aeval ↑x) p) 0) (Eq ((Polynomial.aeval x) p) 0)
  -/
  conv_rhs => rw [← h.eq_iff, map_zero]
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    S : Subalgebra R A
    x : Subtype fun a => Membership.mem S a
    p : Polynomial R
    h : Function.Injective ⇑S.val
    ⊢ Iff (Eq ((Polynomial.aeval ↑x) p) 0) (Eq (S.val ((Polynomial.aeval x) p)) 0)
  -/
  rw [← aeval_algHom_apply, S.val_apply]
  /-
    🎉 no goals
  -/


/-- An algebra is algebraic if and only if it is algebraic as a subalgebra. -/
theorem Algebra.isAlgebraic_iff : Algebra.IsAlgebraic R A ↔ (⊤ : Subalgebra R A).IsAlgebraic := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    ⊢ Iff (Algebra.IsAlgebraic R A) Top.top.IsAlgebraic
  -/
  delta Subalgebra.IsAlgebraic
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    ⊢ Iff (Algebra.IsAlgebraic R A) (∀ (x : A), Membership.mem Top.top x → IsAlgeb …
  -/
  simp only [Algebra.isAlgebraic_def, Algebra.mem_top, forall_prop_of_true]
  /-
    🎉 no goals
  -/


