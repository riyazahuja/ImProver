/--
The logarithmic derivative of a is a′ / a.
-/
def logDeriv : R := a′ / a


@[simp]
lemma logDeriv_zero : logDeriv (0 : R) = 0 := by
  /-
    R : Type u_1
    inst✝¹ : Field R
    inst✝ : Differential R
    ⊢ Eq (Differential.logDeriv 0) 0
  -/
  simp [logDeriv]
  /-
    🎉 no goals
  -/


@[simp]
lemma logDeriv_one : logDeriv (1 : R) = 0 := by
  /-
    R : Type u_1
    inst✝¹ : Field R
    inst✝ : Differential R
    ⊢ Eq (Differential.logDeriv 1) 0
  -/
  simp [logDeriv]
  /-
    🎉 no goals
  -/


lemma logDeriv_mul (ha : a ≠ 0) (hb : b ≠ 0) : logDeriv (a * b) = logDeriv a + logDeriv b := by
  /-
    R : Type u_1
    inst✝¹ : Field R
    inst✝ : Differential R
    a b : R
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (Differential.logDeriv (HMul.hMul a b)) (HAdd.hAdd (Differential.logDeriv …
  -/
  unfold logDeriv
  /-
    R : Type u_1
    inst✝¹ : Field R
    inst✝ : Differential R
    a b : R
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b)′ (HMul.hMul a b)) (HAdd.hAdd (HDiv.hDiv a′ a)  …
  -/
  field_simp
  /-
    R : Type u_1
    inst✝¹ : Field R
    inst✝ : Differential R
    a b : R
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul a b′) (HMul.hMul b a′)) (HAdd.hAdd (HMul.hMul a′ b) …
  -/
  ring
  /-
    🎉 no goals
  -/


lemma logDeriv_div (ha : a ≠ 0) (hb : b ≠ 0) : logDeriv (a / b) = logDeriv a - logDeriv b := by
  /-
    R : Type u_1
    inst✝¹ : Field R
    inst✝ : Differential R
    a b : R
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (Differential.logDeriv (HDiv.hDiv a b)) (HSub.hSub (Differential.logDeriv …
  -/
  unfold logDeriv
  /-
    R : Type u_1
    inst✝¹ : Field R
    inst✝ : Differential R
    a b : R
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv a b)′ (HDiv.hDiv a b)) (HSub.hSub (HDiv.hDiv a′ a)  …
  -/
  field_simp [Derivation.leibniz_div, smul_sub]
  /-
    R : Type u_1
    inst✝¹ : Field R
    inst✝ : Differential R
    a b : R
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (HMul.hMul (HMul.hMul (HSub.hSub (HMul.hMul b a′) (HMul.hMul a b′)) b) (H …
  -/
  ring
  /-
    🎉 no goals
  -/


@[simp]
lemma logDeriv_pow (n : ℕ) (a : R) : logDeriv (a ^ n) = n * logDeriv a := by
  induction n with
  | zero => simp
  | succ n h2 =>
    obtain rfl | hb := eq_or_ne a 0
    · simp
    · rw [Nat.cast_add, Nat.cast_one, add_mul, one_mul, ← h2, pow_succ, logDeriv_mul] <;>
      simp [hb]


lemma logDeriv_eq_zero : logDeriv a = 0 ↔ a′ = 0 :=
              /-
                R : Type u_1
                inst✝¹ : Field R
                inst✝ : Differential R
                a : R
                h : Eq (Differential.logDeriv a) 0
                ⊢ Eq a′ 0
              -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  ⟨fun h ↦ by simp only [logDeriv, div_eq_zero_iff] at h; rcases h with h|h <;> simp [h],
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
             /-
               R : Type u_1
               inst✝¹ : Field R
               inst✝ : Differential R
               a : R
               h : Eq a′ 0
               ⊢ Eq (Differential.logDeriv a) 0
             -/
  fun h ↦ by unfold logDeriv at *; simp [h]⟩
                                   /-
                                     🎉 no goals
                                   -/


lemma logDeriv_multisetProd {ι : Type*} (s : Multiset ι) {f : ι → R} (h : ∀ x ∈ s, f x ≠ 0) :
    logDeriv (s.map f).prod = (s.map fun x ↦ logDeriv (f x)).sum := by
  /-
    R : Type u_1
    inst✝¹ : Field R
    inst✝ : Differential R
    ι : Type u_2
    s : Multiset ι
    f : ι → R
    h : ∀ (x : ι), Membership.mem s x → Ne (f x) 0
    ⊢ Eq (Differential.logDeriv (Multiset.map f s).prod) (Multiset.map (fun x => D …
  -/
  induction s using Multiset.induction_on
    /-
      case empty
      R : Type u_1
      inst✝¹ : Field R
      inst✝ : Differential R
      ι : Type u_2
      f : ι → R
      h : ∀ (x : ι), Membership.mem 0 x → Ne (f x) 0
      ⊢ Eq (Differential.logDeriv (Multiset.map f 0).prod) (Multiset.map (fun x => D …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      R : Type u_1
      inst✝¹ : Field R
      inst✝ : Differential R
      ι : Type u_2
      f : ι → R
      a✝¹ : ι
      s✝ : Multiset ι
      a✝ : (∀ (x : ι), Membership.mem s✝ x → Ne (f x) 0) → Eq (Differential.logDeriv …
      h : ∀ (x : ι), Membership.mem (Multiset.cons a✝¹ s✝) x → Ne (f x) 0
      ⊢ Eq (Differential.logDeriv (Multiset.map f (Multiset.cons a✝¹ s✝)).prod) (Mul …
    -/
  · rename_i h₂
    /-
      case cons
      R : Type u_1
      inst✝¹ : Field R
      inst✝ : Differential R
      ι : Type u_2
      f : ι → R
      a✝ : ι
      s✝ : Multiset ι
      h₂ : (∀ (x : ι), Membership.mem s✝ x → Ne (f x) 0) → Eq (Differential.logDeriv …
      h : ∀ (x : ι), Membership.mem (Multiset.cons a✝ s✝) x → Ne (f x) 0
      ⊢ Eq (Differential.logDeriv (Multiset.map f (Multiset.cons a✝ s✝)).prod) (Mult …
    -/
    simp only [Function.comp_apply, Multiset.map_cons, Multiset.sum_cons, Multiset.prod_cons]
    /-
      case cons
      R : Type u_1
      inst✝¹ : Field R
      inst✝ : Differential R
      ι : Type u_2
      f : ι → R
      a✝ : ι
      s✝ : Multiset ι
      h₂ : (∀ (x : ι), Membership.mem s✝ x → Ne (f x) 0) → Eq (Differential.logDeriv …
      h : ∀ (x : ι), Membership.mem (Multiset.cons a✝ s✝) x → Ne (f x) 0
      ⊢ Eq (Differential.logDeriv (HMul.hMul (f a✝) (Multiset.map f s✝).prod)) (HAdd …
    -/
    rw [← h₂]
      /-
        case cons
        R : Type u_1
        inst✝¹ : Field R
        inst✝ : Differential R
        ι : Type u_2
        f : ι → R
        a✝ : ι
        s✝ : Multiset ι
        h₂ : (∀ (x : ι), Membership.mem s✝ x → Ne (f x) 0) → Eq (Differential.logDeriv …
        h : ∀ (x : ι), Membership.mem (Multiset.cons a✝ s✝) x → Ne (f x) 0
        ⊢ Eq (Differential.logDeriv (HMul.hMul (f a✝) (Multiset.map f s✝).prod)) (HAdd …
      -/
    · apply logDeriv_mul
        /-
          case cons.ha
          R : Type u_1
          inst✝¹ : Field R
          inst✝ : Differential R
          ι : Type u_2
          f : ι → R
          a✝ : ι
          s✝ : Multiset ι
          h₂ : (∀ (x : ι), Membership.mem s✝ x → Ne (f x) 0) → Eq (Differential.logDeriv …
          h : ∀ (x : ι), Membership.mem (Multiset.cons a✝ s✝) x → Ne (f x) 0
          ⊢ Ne (f a✝) 0
        -/
      · simp [h]
        /-
          🎉 no goals
        -/
        /-
          case cons.hb
          R : Type u_1
          inst✝¹ : Field R
          inst✝ : Differential R
          ι : Type u_2
          f : ι → R
          a✝ : ι
          s✝ : Multiset ι
          h₂ : (∀ (x : ι), Membership.mem s✝ x → Ne (f x) 0) → Eq (Differential.logDeriv …
          h : ∀ (x : ι), Membership.mem (Multiset.cons a✝ s✝) x → Ne (f x) 0
          ⊢ Ne (Multiset.map f s✝).prod 0
        -/
      · simp_all
        /-
          🎉 no goals
        -/
      /-
        case cons
        R : Type u_1
        inst✝¹ : Field R
        inst✝ : Differential R
        ι : Type u_2
        f : ι → R
        a✝ : ι
        s✝ : Multiset ι
        h₂ : (∀ (x : ι), Membership.mem s✝ x → Ne (f x) 0) → Eq (Differential.logDeriv …
        h : ∀ (x : ι), Membership.mem (Multiset.cons a✝ s✝) x → Ne (f x) 0
        ⊢ ∀ (x : ι), Membership.mem s✝ x → Ne (f x) 0
      -/
    · simp_all
      /-
        🎉 no goals
      -/


lemma logDeriv_prod (ι : Type*) (s : Finset ι) (f : ι → R) (h : ∀ x ∈ s, f x ≠ 0) :
    logDeriv (∏ x ∈ s, f x) = ∑ x ∈ s, logDeriv (f x) := logDeriv_multisetProd _ h


lemma logDeriv_prod_of_eq_zero (ι : Type*) (s : Finset ι) (f : ι → R) (h : ∀ x ∈ s, f x = 0) :
    logDeriv (∏ x ∈ s, f x) = ∑ x ∈ s, logDeriv (f x) := by
  /-
    R : Type u_1
    inst✝¹ : Field R
    inst✝ : Differential R
    ι : Type u_2
    s : Finset ι
    f : ι → R
    h : ∀ (x : ι), Membership.mem s x → Eq (f x) 0
    ⊢ Eq (Differential.logDeriv (s.prod fun x => f x)) (s.sum fun x => Differentia …
  -/
  unfold logDeriv
  /-
    R : Type u_1
    inst✝¹ : Field R
    inst✝ : Differential R
    ι : Type u_2
    s : Finset ι
    f : ι → R
    h : ∀ (x : ι), Membership.mem s x → Eq (f x) 0
    ⊢ Eq (HDiv.hDiv (s.prod fun x => f x)′ (s.prod fun x => f x)) (s.sum fun x =>  …
  -/
  simp_all
  /-
    🎉 no goals
  -/


lemma logDeriv_algebraMap {F K : Type*} [Field F] [Field K] [Differential F] [Differential K]
    [Algebra F K] [DifferentialAlgebra F K]
    (a : F) : logDeriv (algebraMap F K a) = algebraMap F K (logDeriv a) := by
  /-
    F : Type u_2
    K : Type u_3
    inst✝⁵ : Field F
    inst✝⁴ : Field K
    inst✝³ : Differential F
    inst✝² : Differential K
    inst✝¹ : Algebra F K
    inst✝ : DifferentialAlgebra F K
    a : F
    ⊢ Eq (Differential.logDeriv ((algebraMap F K) a)) ((algebraMap F K) (Different …
  -/
  unfold logDeriv
  /-
    F : Type u_2
    K : Type u_3
    inst✝⁵ : Field F
    inst✝⁴ : Field K
    inst✝³ : Differential F
    inst✝² : Differential K
    inst✝¹ : Algebra F K
    inst✝ : DifferentialAlgebra F K
    a : F
    ⊢ Eq (HDiv.hDiv ((algebraMap F K) a)′ ((algebraMap F K) a)) ((algebraMap F K)  …
  -/
  simp [deriv_algebraMap]
  /-
    🎉 no goals
  -/


@[norm_cast]
lemma _root_.algebraMap.coe_logDeriv {F K : Type*} [Field F] [Field K] [Differential F]
    [Differential K] [Algebra F K] [DifferentialAlgebra F K]
    (a : F) : logDeriv a = logDeriv (a : K) := (logDeriv_algebraMap a).symm


