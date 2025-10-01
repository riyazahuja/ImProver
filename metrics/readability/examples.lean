import ImProver.metrics.tagger
import Mathlib.Analysis.Calculus.Darboux
import Mathlib.Probability.StrongLaw

open Set

variable {α : Type*} (s t u : Set α)

@[improver_example set_union, version unoptimized]
example : s \ t ∪ t = s ∪ t := by
  ext x
  constructor
  · intro hyp
    cases hyp
    case inl h' =>
      left
      exact h'.left
    case inr h' =>
      right
      exact h'
  · intro hyp
    cases hyp
    case inl h' =>
      by_cases h : x ∈ t
      · right
        exact h
      · left
        exact ⟨h', h⟩
    case inr h' =>
      right
      exact h'

@[improver_example set_union, version optimized]
example : s \ t ∪ t = s ∪ t := by
  -- Prove by extensionality; break up the proof into a forward and backward direction
  ext x
  constructor
  · -- Since x ∈ (s \ t) ∪ t, we know x is in either s or t
    rintro (⟨xs, _⟩ | xt)
    · left
      exact xs
    · right
      exact xt
  -- Case on whether x is in t; if it is then we're done, otherwise it must be in s
  by_cases h : x ∈ t
  · intro
    right
    exact h
  rintro (xs | xt)
  · left
    use xs
  contradiction




open Function
variable {ι α β M N P : Type*} {G : Type*} {H : Type*} {F : Type*}

@[improver_example homomorphisms_cancel_left, version unoptimized]
theorem cancel_left [Mul M] [Mul N] [Mul P] {g : N →ₙ* P} {f₁ f₂ : M →ₙ* N}
    (hg : Function.Injective g) : g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
  ⟨fun h => MulHom.ext fun x => hg <| by rw [← MulHom.comp_apply, h, MulHom.comp_apply],
    fun h => h ▸ rfl⟩

@[improver_example homomorphisms_cancel_left, version optimized]
theorem cancel_left' [Mul M] [Mul N] [Mul P] {g : N →ₙ* P} {f₁ f₂ : M →ₙ* N}
    (hg : Function.Injective g) : g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ := by
  constructor -- Split the proof into forward and backward directions
  · intro h; apply MulHom.ext; intro x; -- Function extensionality to show equality of f₁ and f₂ (keeping it short since original proof was only two lines)
    apply hg; simp_rw [← MulHom.comp_apply, h] -- Since g is injective, we can apply it to both sides
  aesop -- Other way around's pretty trivial, just apply g on both sides


open Filter Set

variable {a b : ℝ} {f f' : ℝ → ℝ}


/-- **Darboux's theorem**: if `a ≤ b` and `f' b < m < f' a`, then `f' c = m` for some `c ∈ (a, b)`.
-/
@[improver_example darboux_theorem, version unoptimized]
theorem darboux (hab : a ≤ b)
    (hf : ∀ x ∈ Icc a b, HasDerivWithinAt f (f' x) (Icc a b) x) {m : ℝ} (hma : m < f' a)
    (hmb : f' b < m) : m ∈ f' '' Ioo a b :=
  let ⟨c, cmem, hc⟩ :=
    exists_hasDerivWithinAt_eq_of_gt_of_lt hab (fun x hx => (hf x hx).neg) (neg_lt_neg hma)
      (neg_lt_neg hmb)
  ⟨c, cmem, neg_injective hc⟩

/-- **Darboux's theorem**: if `a ≤ b` and `f' b < m < f' a`, then `f' c = m` for some `c ∈ (a, b)`.
-/
@[improver_example darboux_theorem, version optimized]
theorem darboux' (hab : a ≤ b)
    (hf : ∀ x ∈ Icc a b, HasDerivWithinAt f (f' x) (Icc a b) x) {m : ℝ} (hma : m < f' a)
    (hmb : f' b < m) : m ∈ f' '' Ioo a b := by
  have deriv_of_neg_eq_neg_of_deriv : ∀ x ∈ Icc a b, HasDerivWithinAt (fun x => -f x) (-f' x) (Icc a b) x := by
    intro x hx; simpa using (hf x hx).neg -- At all points in [a, b], the derivative of the negated function is the negation of the derivative
  rcases exists_hasDerivWithinAt_eq_of_gt_of_lt hab deriv_of_neg_eq_neg_of_deriv (by linarith)
      (neg_lt_neg hmb) with ⟨c, cmem, hc⟩ -- Apply the reversed version of Darboux's theorem to the negated function
  exact ⟨c, cmem, neg_injective hc⟩


noncomputable section
namespace ProbabilityTheory
open MeasureTheory

@[improver_example truncate_zero, version unoptimized]
theorem truncate_zero (f : α → ℝ) : truncation f 0 = 0 := by simp [truncation]; rfl

@[improver_example truncate_zero, version optimized]
theorem truncate_zero' (f : α → ℝ) : truncation f 0 = 0 := by simp [truncation]; rfl -- Proof is trivial and thus already easy to read; no optimization needed
end ProbabilityTheory
