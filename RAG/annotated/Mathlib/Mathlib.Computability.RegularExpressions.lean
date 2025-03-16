/-- This is the definition of regular expressions. The names used here is to mirror the definition
of a Kleene algebra (https://en.wikipedia.org/wiki/Kleene_algebra).
* `0` (`zero`) matches nothing
* `1` (`epsilon`) matches only the empty string
* `char a` matches only the string 'a'
* `star P` matches any finite concatenation of strings which match `P`
* `P + Q` (`plus P Q`) matches anything which match `P` or `Q`
* `P * Q` (`comp P Q`) matches `x ++ y` if `x` matches `P` and `y` matches `Q`
-/
inductive RegularExpression (α : Type u) : Type u
  | zero : RegularExpression α
  | epsilon : RegularExpression α
  | char : α → RegularExpression α
  | plus : RegularExpression α → RegularExpression α → RegularExpression α
  | comp : RegularExpression α → RegularExpression α → RegularExpression α
  | star : RegularExpression α → RegularExpression α


-- Porting note: `simpNF` gets grumpy about how the `foo_def`s below can simplify these..

instance : Inhabited (RegularExpression α) :=
  ⟨zero⟩


instance : Add (RegularExpression α) :=
  ⟨plus⟩


instance : Mul (RegularExpression α) :=
  ⟨comp⟩


instance : One (RegularExpression α) :=
  ⟨epsilon⟩


instance : Zero (RegularExpression α) :=
  ⟨zero⟩


instance : Pow (RegularExpression α) ℕ :=
  ⟨fun n r => npowRec r n⟩

-- Porting note: declaration in an imported module
--attribute [match_pattern] Mul.mul


@[simp]
theorem zero_def : (zero : RegularExpression α) = 0 :=
  rfl


@[simp]
theorem one_def : (epsilon : RegularExpression α) = 1 :=
  rfl


@[simp]
theorem plus_def (P Q : RegularExpression α) : plus P Q = P + Q :=
  rfl


@[simp]
theorem comp_def (P Q : RegularExpression α) : comp P Q = P * Q :=
  rfl

-- Porting note: `matches` is reserved, moved to `matches'`

/-- `matches' P` provides a language which contains all strings that `P` matches -/
-- Porting note: was '@[simp] but removed based on
-- https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/simpNF.20issues.20in.20Computability.2ERegularExpressions.20!4.232306/near/328355362
def matches' : RegularExpression α → Language α
  | 0 => 0
  | 1 => 1
  | char a => {[a]}
  | P + Q => P.matches' + Q.matches'
  | comp P Q => P.matches' * Q.matches'
  | star P => P.matches'∗


@[simp]
theorem matches'_zero : (0 : RegularExpression α).matches' = 0 :=
  rfl


@[simp]
theorem matches'_epsilon : (1 : RegularExpression α).matches' = 1 :=
  rfl


@[simp]
theorem matches'_char (a : α) : (char a).matches' = {[a]} :=
  rfl


@[simp]
theorem matches'_add (P Q : RegularExpression α) : (P + Q).matches' = P.matches' + Q.matches' :=
  rfl


@[simp]
theorem matches'_mul (P Q : RegularExpression α) : (P * Q).matches' = P.matches' * Q.matches' :=
  rfl


@[simp]
theorem matches'_pow (P : RegularExpression α) : ∀ n : ℕ, (P ^ n).matches' = P.matches' ^ n
  | 0 => matches'_epsilon
  | n + 1 => (matches'_mul _ _).trans <| Eq.trans
      (congrFun (congrArg HMul.hMul (matches'_pow P n)) (matches' P))
      (pow_succ _ n).symm


@[simp]
theorem matches'_star (P : RegularExpression α) : P.star.matches' = P.matches'∗ :=
  rfl


/-- `matchEpsilon P` is true if and only if `P` matches the empty string -/
def matchEpsilon : RegularExpression α → Bool
  | 0 => false
  | 1 => true
  | char _ => false
  | P + Q => P.matchEpsilon || Q.matchEpsilon
  | comp P Q => P.matchEpsilon && Q.matchEpsilon
  | star _P => true


/-- `P.deriv a` matches `x` if `P` matches `a :: x`, the Brzozowski derivative of `P` with respect
  to `a` -/
def deriv : RegularExpression α → α → RegularExpression α
  | 0, _ => 0
  | 1, _ => 0
  | char a₁, a₂ => if a₁ = a₂ then 1 else 0
  | P + Q, a => deriv P a + deriv Q a
  | comp P Q, a => if P.matchEpsilon then deriv P a * Q + deriv Q a else deriv P a * Q
  | star P, a => deriv P a * star P


@[simp]
theorem deriv_zero (a : α) : deriv 0 a = 0 :=
  rfl


@[simp]
theorem deriv_one (a : α) : deriv 1 a = 0 :=
  rfl


@[simp]
theorem deriv_char_self (a : α) : deriv (char a) a = 1 :=
  if_pos rfl


@[simp]
theorem deriv_char_of_ne (h : a ≠ b) : deriv (char a) b = 0 :=
  if_neg h


@[simp]
theorem deriv_add (P Q : RegularExpression α) (a : α) : deriv (P + Q) a = deriv P a + deriv Q a :=
  rfl


@[simp]
theorem deriv_star (P : RegularExpression α) (a : α) : deriv P.star a = deriv P a * star P :=
  rfl


/-- `P.rmatch x` is true if and only if `P` matches `x`. This is a computable definition equivalent
  to `matches'`. -/
def rmatch : RegularExpression α → List α → Bool
  | P, [] => matchEpsilon P
  | P, a :: as => rmatch (P.deriv a) as


@[simp]
theorem zero_rmatch (x : List α) : rmatch 0 x = false := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x : List α
    ⊢ Eq (RegularExpression.rmatch 0 x) Bool.false
  -/
                  /-
                    🎉 no goals
                  -/
  induction x <;> simp [rmatch, matchEpsilon, *]
                  /-
                    🎉 no goals
                  -/


theorem one_rmatch_iff (x : List α) : rmatch 1 x ↔ x = [] := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x : List α
    ⊢ Iff (Eq (RegularExpression.rmatch 1 x) Bool.true) (Eq x List.nil)
  -/
                  /-
                    🎉 no goals
                  -/
  induction x <;> simp [rmatch, matchEpsilon, *]
                  /-
                    🎉 no goals
                  -/


theorem char_rmatch_iff (a : α) (x : List α) : rmatch (char a) x ↔ x = [a] := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    x : List α
    ⊢ Iff (Eq ((RegularExpression.char a).rmatch x) Bool.true) (Eq x (List.cons a  …
  -/
  cases' x with _ x
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      ⊢ Iff (Eq ((RegularExpression.char a).rmatch List.nil) Bool.true) (Eq List.nil …
    -/
  · exact of_decide_eq_true rfl
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    inst✝ : DecidableEq α
    a head✝ : α
    x : List α
    ⊢ Iff (Eq ((RegularExpression.char a).rmatch (List.cons head✝ x)) Bool.true) ( …
  -/
  cases' x with head tail
    /-
      case cons.nil
      α : Type u_1
      inst✝ : DecidableEq α
      a head✝ : α
      ⊢ Iff (Eq ((RegularExpression.char a).rmatch (List.cons head✝ List.nil)) Bool. …
    -/
  · rw [rmatch, deriv]
    /-
      case cons.nil
      α : Type u_1
      inst✝ : DecidableEq α
      a head✝ : α
      ⊢ Iff (Eq ((ite (Eq a head✝) 1 0).rmatch List.nil) Bool.true) (Eq (List.cons h …
    -/
    split_ifs
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        a head✝ : α
        h✝ : Eq a head✝
        ⊢ Iff (Eq (RegularExpression.rmatch 1 List.nil) Bool.true) (Eq (List.cons head …
      -/
    · tauto
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        a head✝ : α
        h✝ : Not (Eq a head✝)
        ⊢ Iff (Eq (RegularExpression.rmatch 0 List.nil) Bool.true) (Eq (List.cons head …
      -/
    · simp [List.singleton_inj]; tauto
                                 /-
                                   🎉 no goals
                                 -/
    /-
      case cons.cons
      α : Type u_1
      inst✝ : DecidableEq α
      a head✝ head : α
      tail : List α
      ⊢ Iff (Eq ((RegularExpression.char a).rmatch (List.cons head✝ (List.cons head  …
    -/
  · rw [rmatch, rmatch, deriv]
    /-
      case cons.cons
      α : Type u_1
      inst✝ : DecidableEq α
      a head✝ head : α
      tail : List α
      ⊢ Iff (Eq (((ite (Eq a head✝) 1 0).deriv head).rmatch tail) Bool.true) (Eq (Li …
    -/
    split_ifs with h
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        a head✝ head : α
        tail : List α
        h : Eq a head✝
        ⊢ Iff (Eq ((RegularExpression.deriv 1 head).rmatch tail) Bool.true) (Eq (List. …
      -/
    · simp only [deriv_one, zero_rmatch, cons.injEq, and_false, reduceCtorEq]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        a head✝ head : α
        tail : List α
        h : Not (Eq a head✝)
        ⊢ Iff (Eq ((RegularExpression.deriv 0 head).rmatch tail) Bool.true) (Eq (List. …
      -/
    · simp only [deriv_zero, zero_rmatch, cons.injEq, and_false, reduceCtorEq]
      /-
        🎉 no goals
      -/


theorem add_rmatch_iff (P Q : RegularExpression α) (x : List α) :
    (P + Q).rmatch x ↔ P.rmatch x ∨ Q.rmatch x := by
  induction x generalizing P Q with
  | nil => simp only [rmatch, matchEpsilon, Bool.or_eq_true_iff]
  | cons _ _ ih =>
    repeat rw [rmatch]
    rw [deriv_add]
    exact ih _ _


theorem mul_rmatch_iff (P Q : RegularExpression α) (x : List α) :
    (P * Q).rmatch x ↔ ∃ t u : List α, x = t ++ u ∧ P.rmatch t ∧ Q.rmatch u := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    P Q : RegularExpression α
    x : List α
    ⊢ Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.true) (Exists fun t => Exists fun u  …
  -/
  induction' x with a x ih generalizing P Q
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      P Q : RegularExpression α
      ⊢ Iff (Eq ((HMul.hMul P Q).rmatch List.nil) Bool.true) (Exists fun t => Exists …
    -/
  · rw [rmatch]; simp only [matchEpsilon]
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      P Q : RegularExpression α
      ⊢ Iff (Eq (P.matchEpsilon.and Q.matchEpsilon) Bool.true) (Exists fun t => Exis …
    -/
    constructor
      /-
        case nil.mp
        α : Type u_1
        inst✝ : DecidableEq α
        P Q : RegularExpression α
        ⊢ Eq (P.matchEpsilon.and Q.matchEpsilon) Bool.true → Exists fun t => Exists fu …
      -/
    · intro h
      /-
        case nil.mp
        α : Type u_1
        inst✝ : DecidableEq α
        P Q : RegularExpression α
        h : Eq (P.matchEpsilon.and Q.matchEpsilon) Bool.true
        ⊢ Exists fun t => Exists fun u => And (Eq List.nil (HAppend.hAppend t u)) (And …
      -/
      refine ⟨[], [], rfl, ?_⟩
      /-
        case nil.mp
        α : Type u_1
        inst✝ : DecidableEq α
        P Q : RegularExpression α
        h : Eq (P.matchEpsilon.and Q.matchEpsilon) Bool.true
        ⊢ And (Eq (P.rmatch List.nil) Bool.true) (Eq (Q.rmatch List.nil) Bool.true)
      -/
      rw [rmatch, rmatch]
      /-
        case nil.mp
        α : Type u_1
        inst✝ : DecidableEq α
        P Q : RegularExpression α
        h : Eq (P.matchEpsilon.and Q.matchEpsilon) Bool.true
        ⊢ And (Eq P.matchEpsilon Bool.true) (Eq Q.matchEpsilon Bool.true)
      -/
      rwa [Bool.and_eq_true_iff] at h
      /-
        🎉 no goals
      -/
      /-
        case nil.mpr
        α : Type u_1
        inst✝ : DecidableEq α
        P Q : RegularExpression α
        ⊢ (Exists fun t => Exists fun u => And (Eq List.nil (HAppend.hAppend t u)) (An …
      -/
    · rintro ⟨t, u, h₁, h₂⟩
      /-
        case nil.mpr.intro.intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        P Q : RegularExpression α
        t u : List α
        h₁ : Eq List.nil (HAppend.hAppend t u)
        h₂ : And (Eq (P.rmatch t) Bool.true) (Eq (Q.rmatch u) Bool.true)
        ⊢ Eq (P.matchEpsilon.and Q.matchEpsilon) Bool.true
      -/
      cases' List.append_eq_nil.1 h₁.symm with ht hu
      /-
        case nil.mpr.intro.intro.intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        P Q : RegularExpression α
        t u : List α
        h₁ : Eq List.nil (HAppend.hAppend t u)
        h₂ : And (Eq (P.rmatch t) Bool.true) (Eq (Q.rmatch u) Bool.true)
        ht : Eq t List.nil
        hu : Eq u List.nil
        ⊢ Eq (P.matchEpsilon.and Q.matchEpsilon) Bool.true
      -/
      subst ht
      /-
        case nil.mpr.intro.intro.intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        P Q : RegularExpression α
        u : List α
        hu : Eq u List.nil
        h₁ : Eq List.nil (HAppend.hAppend List.nil u)
        h₂ : And (Eq (P.rmatch List.nil) Bool.true) (Eq (Q.rmatch u) Bool.true)
        ⊢ Eq (P.matchEpsilon.and Q.matchEpsilon) Bool.true
      -/
      subst hu
      /-
        case nil.mpr.intro.intro.intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        P Q : RegularExpression α
        h₁ : Eq List.nil (HAppend.hAppend List.nil List.nil)
        h₂ : And (Eq (P.rmatch List.nil) Bool.true) (Eq (Q.rmatch List.nil) Bool.true)
        ⊢ Eq (P.matchEpsilon.and Q.matchEpsilon) Bool.true
      -/
      repeat rw [rmatch] at h₂
      /-
        case nil.mpr.intro.intro.intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        P Q : RegularExpression α
        h₁ : Eq List.nil (HAppend.hAppend List.nil List.nil)
        h₂ : And (Eq P.matchEpsilon Bool.true) (Eq Q.matchEpsilon Bool.true)
        ⊢ Eq (P.matchEpsilon.and Q.matchEpsilon) Bool.true
      -/
      simp [h₂]
      /-
        🎉 no goals
      -/
    /-
      case cons
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      x : List α
      ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
      P Q : RegularExpression α
      ⊢ Iff (Eq ((HMul.hMul P Q).rmatch (List.cons a x)) Bool.true) (Exists fun t => …
    -/
  · rw [rmatch]; simp only [deriv]
    /-
      case cons
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      x : List α
      ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
      P Q : RegularExpression α
      ⊢ Iff (Eq ((ite (Eq P.matchEpsilon Bool.true) (HAdd.hAdd (HMul.hMul (P.deriv a …
    -/
    split_ifs with hepsilon
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        x : List α
        ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
        P Q : RegularExpression α
        hepsilon : Eq P.matchEpsilon Bool.true
        ⊢ Iff (Eq ((HAdd.hAdd (HMul.hMul (P.deriv a) Q) (Q.deriv a)).rmatch x) Bool.tr …
      -/
    · rw [add_rmatch_iff, ih]
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        x : List α
        ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
        P Q : RegularExpression α
        hepsilon : Eq P.matchEpsilon Bool.true
        ⊢ Iff (Or (Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (A …
      -/
      constructor
        /-
          case pos.mp
          α : Type u_1
          inst✝ : DecidableEq α
          a : α
          x : List α
          ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
          P Q : RegularExpression α
          hepsilon : Eq P.matchEpsilon Bool.true
          ⊢ Or (Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (E …
        -/
      · rintro (⟨t, u, _⟩ | h)
          /-
            case pos.mp.inl.intro.intro
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Eq P.matchEpsilon Bool.true
            t u : List α
            h✝ : And (Eq x (HAppend.hAppend t u)) (And (Eq ((P.deriv a).rmatch t) Bool.tru …
            ⊢ Exists fun t => Exists fun u => And (Eq (List.cons a x) (HAppend.hAppend t u …
          -/
        · exact ⟨a :: t, u, by tauto⟩
          /-
            🎉 no goals
          -/
          /-
            case pos.mp.inr
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Eq P.matchEpsilon Bool.true
            h : Eq ((Q.deriv a).rmatch x) Bool.true
            ⊢ Exists fun t => Exists fun u => And (Eq (List.cons a x) (HAppend.hAppend t u …
          -/
        · exact ⟨[], a :: x, rfl, hepsilon, h⟩
          /-
            🎉 no goals
          -/
        /-
          case pos.mpr
          α : Type u_1
          inst✝ : DecidableEq α
          a : α
          x : List α
          ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
          P Q : RegularExpression α
          hepsilon : Eq P.matchEpsilon Bool.true
          ⊢ (Exists fun t => Exists fun u => And (Eq (List.cons a x) (HAppend.hAppend t  …
        -/
      · rintro ⟨t, u, h, hP, hQ⟩
        /-
          case pos.mpr.intro.intro.intro.intro
          α : Type u_1
          inst✝ : DecidableEq α
          a : α
          x : List α
          ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
          P Q : RegularExpression α
          hepsilon : Eq P.matchEpsilon Bool.true
          t u : List α
          h : Eq (List.cons a x) (HAppend.hAppend t u)
          hP : Eq (P.rmatch t) Bool.true
          hQ : Eq (Q.rmatch u) Bool.true
          ⊢ Or (Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (E …
        -/
        cases' t with b t
          /-
            case pos.mpr.intro.intro.intro.intro.nil
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Eq P.matchEpsilon Bool.true
            u : List α
            hQ : Eq (Q.rmatch u) Bool.true
            h : Eq (List.cons a x) (HAppend.hAppend List.nil u)
            hP : Eq (P.rmatch List.nil) Bool.true
            ⊢ Or (Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (E …
          -/
        · right
          /-
            case pos.mpr.intro.intro.intro.intro.nil.h
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Eq P.matchEpsilon Bool.true
            u : List α
            hQ : Eq (Q.rmatch u) Bool.true
            h : Eq (List.cons a x) (HAppend.hAppend List.nil u)
            hP : Eq (P.rmatch List.nil) Bool.true
            ⊢ Eq ((Q.deriv a).rmatch x) Bool.true
          -/
          rw [List.nil_append] at h
          /-
            case pos.mpr.intro.intro.intro.intro.nil.h
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Eq P.matchEpsilon Bool.true
            u : List α
            hQ : Eq (Q.rmatch u) Bool.true
            h : Eq (List.cons a x) u
            hP : Eq (P.rmatch List.nil) Bool.true
            ⊢ Eq ((Q.deriv a).rmatch x) Bool.true
          -/
          rw [← h] at hQ
          /-
            case pos.mpr.intro.intro.intro.intro.nil.h
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Eq P.matchEpsilon Bool.true
            u : List α
            hQ : Eq (Q.rmatch (List.cons a x)) Bool.true
            h : Eq (List.cons a x) u
            hP : Eq (P.rmatch List.nil) Bool.true
            ⊢ Eq ((Q.deriv a).rmatch x) Bool.true
          -/
          exact hQ
          /-
            🎉 no goals
          -/
          /-
            case pos.mpr.intro.intro.intro.intro.cons
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Eq P.matchEpsilon Bool.true
            u : List α
            hQ : Eq (Q.rmatch u) Bool.true
            b : α
            t : List α
            h : Eq (List.cons a x) (HAppend.hAppend (List.cons b t) u)
            hP : Eq (P.rmatch (List.cons b t)) Bool.true
            ⊢ Or (Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (E …
          -/
        · left
          /-
            case pos.mpr.intro.intro.intro.intro.cons.h
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Eq P.matchEpsilon Bool.true
            u : List α
            hQ : Eq (Q.rmatch u) Bool.true
            b : α
            t : List α
            h : Eq (List.cons a x) (HAppend.hAppend (List.cons b t) u)
            hP : Eq (P.rmatch (List.cons b t)) Bool.true
            ⊢ Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (Eq (( …
          -/
          rw [List.cons_append, List.cons_eq_cons] at h
          /-
            case pos.mpr.intro.intro.intro.intro.cons.h
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Eq P.matchEpsilon Bool.true
            u : List α
            hQ : Eq (Q.rmatch u) Bool.true
            b : α
            t : List α
            h : And (Eq a b) (Eq x (HAppend.hAppend t u))
            hP : Eq (P.rmatch (List.cons b t)) Bool.true
            ⊢ Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (Eq (( …
          -/
          refine ⟨t, u, h.2, ?_, hQ⟩
          /-
            case pos.mpr.intro.intro.intro.intro.cons.h
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Eq P.matchEpsilon Bool.true
            u : List α
            hQ : Eq (Q.rmatch u) Bool.true
            b : α
            t : List α
            h : And (Eq a b) (Eq x (HAppend.hAppend t u))
            hP : Eq (P.rmatch (List.cons b t)) Bool.true
            ⊢ Eq ((P.deriv a).rmatch t) Bool.true
          -/
          rw [rmatch] at hP
          /-
            case pos.mpr.intro.intro.intro.intro.cons.h
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Eq P.matchEpsilon Bool.true
            u : List α
            hQ : Eq (Q.rmatch u) Bool.true
            b : α
            t : List α
            h : And (Eq a b) (Eq x (HAppend.hAppend t u))
            hP : Eq ((P.deriv b).rmatch t) Bool.true
            ⊢ Eq ((P.deriv a).rmatch t) Bool.true
          -/
          convert hP
          /-
            case h.e'_2.h.e'_3.h.e'_4
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Eq P.matchEpsilon Bool.true
            u : List α
            hQ : Eq (Q.rmatch u) Bool.true
            b : α
            t : List α
            h : And (Eq a b) (Eq x (HAppend.hAppend t u))
            hP : Eq ((P.deriv b).rmatch t) Bool.true
            ⊢ Eq a b
          -/
          exact h.1
          /-
            🎉 no goals
          -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        x : List α
        ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
        P Q : RegularExpression α
        hepsilon : Not (Eq P.matchEpsilon Bool.true)
        ⊢ Iff (Eq ((HMul.hMul (P.deriv a) Q).rmatch x) Bool.true) (Exists fun t => Exi …
      -/
    · rw [ih]
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        x : List α
        ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
        P Q : RegularExpression α
        hepsilon : Not (Eq P.matchEpsilon Bool.true)
        ⊢ Iff (Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And ( …
      -/
      constructor <;> rintro ⟨t, u, h, hP, hQ⟩
        /-
          case neg.mp.intro.intro.intro.intro
          α : Type u_1
          inst✝ : DecidableEq α
          a : α
          x : List α
          ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
          P Q : RegularExpression α
          hepsilon : Not (Eq P.matchEpsilon Bool.true)
          t u : List α
          h : Eq x (HAppend.hAppend t u)
          hP : Eq ((P.deriv a).rmatch t) Bool.true
          hQ : Eq (Q.rmatch u) Bool.true
          ⊢ Exists fun t => Exists fun u => And (Eq (List.cons a x) (HAppend.hAppend t u …
        -/
      · exact ⟨a :: t, u, by tauto⟩
        /-
          🎉 no goals
        -/
        /-
          case neg.mpr.intro.intro.intro.intro
          α : Type u_1
          inst✝ : DecidableEq α
          a : α
          x : List α
          ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
          P Q : RegularExpression α
          hepsilon : Not (Eq P.matchEpsilon Bool.true)
          t u : List α
          h : Eq (List.cons a x) (HAppend.hAppend t u)
          hP : Eq (P.rmatch t) Bool.true
          hQ : Eq (Q.rmatch u) Bool.true
          ⊢ Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (Eq (( …
        -/
      · cases' t with b t
          /-
            case neg.mpr.intro.intro.intro.intro.nil
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Not (Eq P.matchEpsilon Bool.true)
            u : List α
            hQ : Eq (Q.rmatch u) Bool.true
            h : Eq (List.cons a x) (HAppend.hAppend List.nil u)
            hP : Eq (P.rmatch List.nil) Bool.true
            ⊢ Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (Eq (( …
          -/
        · contradiction
          /-
            🎉 no goals
          -/
          /-
            case neg.mpr.intro.intro.intro.intro.cons
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Not (Eq P.matchEpsilon Bool.true)
            u : List α
            hQ : Eq (Q.rmatch u) Bool.true
            b : α
            t : List α
            h : Eq (List.cons a x) (HAppend.hAppend (List.cons b t) u)
            hP : Eq (P.rmatch (List.cons b t)) Bool.true
            ⊢ Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (Eq (( …
          -/
        · rw [List.cons_append, List.cons_eq_cons] at h
          /-
            case neg.mpr.intro.intro.intro.intro.cons
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Not (Eq P.matchEpsilon Bool.true)
            u : List α
            hQ : Eq (Q.rmatch u) Bool.true
            b : α
            t : List α
            h : And (Eq a b) (Eq x (HAppend.hAppend t u))
            hP : Eq (P.rmatch (List.cons b t)) Bool.true
            ⊢ Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (Eq (( …
          -/
          refine ⟨t, u, h.2, ?_, hQ⟩
          /-
            case neg.mpr.intro.intro.intro.intro.cons
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Not (Eq P.matchEpsilon Bool.true)
            u : List α
            hQ : Eq (Q.rmatch u) Bool.true
            b : α
            t : List α
            h : And (Eq a b) (Eq x (HAppend.hAppend t u))
            hP : Eq (P.rmatch (List.cons b t)) Bool.true
            ⊢ Eq ((P.deriv a).rmatch t) Bool.true
          -/
          rw [rmatch] at hP
          /-
            case neg.mpr.intro.intro.intro.intro.cons
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Not (Eq P.matchEpsilon Bool.true)
            u : List α
            hQ : Eq (Q.rmatch u) Bool.true
            b : α
            t : List α
            h : And (Eq a b) (Eq x (HAppend.hAppend t u))
            hP : Eq ((P.deriv b).rmatch t) Bool.true
            ⊢ Eq ((P.deriv a).rmatch t) Bool.true
          -/
          convert hP
          /-
            case h.e'_2.h.e'_3.h.e'_4
            α : Type u_1
            inst✝ : DecidableEq α
            a : α
            x : List α
            ih : ∀ (P Q : RegularExpression α), Iff (Eq ((HMul.hMul P Q).rmatch x) Bool.tr …
            P Q : RegularExpression α
            hepsilon : Not (Eq P.matchEpsilon Bool.true)
            u : List α
            hQ : Eq (Q.rmatch u) Bool.true
            b : α
            t : List α
            h : And (Eq a b) (Eq x (HAppend.hAppend t u))
            hP : Eq ((P.deriv b).rmatch t) Bool.true
            ⊢ Eq a b
          -/
          exact h.1
          /-
            🎉 no goals
          -/


theorem star_rmatch_iff (P : RegularExpression α) :
    ∀ x : List α, (star P).rmatch x ↔ ∃ S : List (List α), x
          = S.flatten ∧ ∀ t ∈ S, t ≠ [] ∧ P.rmatch t :=
  fun x => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      P : RegularExpression α
      x : List α
      ⊢ Iff (Eq (P.star.rmatch x) Bool.true) (Exists fun S => And (Eq x S.flatten) ( …
    -/
    have IH := fun t (_h : List.length t < List.length x) => star_rmatch_iff P t
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      P : RegularExpression α
      x : List α
      IH : ∀ (t : List α), LT.lt t.length x.length → Iff (Eq (P.star.rmatch t) Bool. …
      ⊢ Iff (Eq (P.star.rmatch x) Bool.true) (Exists fun S => And (Eq x S.flatten) ( …
    -/
    clear star_rmatch_iff
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      P : RegularExpression α
      x : List α
      IH : ∀ (t : List α), LT.lt t.length x.length → Iff (Eq (P.star.rmatch t) Bool. …
      ⊢ Iff (Eq (P.star.rmatch x) Bool.true) (Exists fun S => And (Eq x S.flatten) ( …
    -/
    constructor
      /-
        case mp
        α : Type u_1
        inst✝ : DecidableEq α
        P : RegularExpression α
        x : List α
        IH : ∀ (t : List α), LT.lt t.length x.length → Iff (Eq (P.star.rmatch t) Bool. …
        ⊢ Eq (P.star.rmatch x) Bool.true → Exists fun S => And (Eq x S.flatten) (∀ (t  …
      -/
    · cases' x with a x
        /-
          case mp.nil
          α : Type u_1
          inst✝ : DecidableEq α
          P : RegularExpression α
          IH : ∀ (t : List α), LT.lt t.length List.nil.length → Iff (Eq (P.star.rmatch t …
          ⊢ Eq (P.star.rmatch List.nil) Bool.true → Exists fun S => And (Eq List.nil S.f …
        -/
      · intro _h
        /-
          case mp.nil
          α : Type u_1
          inst✝ : DecidableEq α
          P : RegularExpression α
          IH : ∀ (t : List α), LT.lt t.length List.nil.length → Iff (Eq (P.star.rmatch t …
          _h : Eq (P.star.rmatch List.nil) Bool.true
          ⊢ Exists fun S => And (Eq List.nil S.flatten) (∀ (t : List α), Membership.mem  …
        -/
        use []; dsimp; tauto
                       /-
                         🎉 no goals
                       -/
        /-
          case mp.cons
          α : Type u_1
          inst✝ : DecidableEq α
          P : RegularExpression α
          a : α
          x : List α
          IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
          ⊢ Eq (P.star.rmatch (List.cons a x)) Bool.true → Exists fun S => And (Eq (List …
        -/
      · rw [rmatch, deriv, mul_rmatch_iff]
        /-
          case mp.cons
          α : Type u_1
          inst✝ : DecidableEq α
          P : RegularExpression α
          a : α
          x : List α
          IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
          ⊢ (Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (Eq ( …
        -/
        rintro ⟨t, u, hs, ht, hu⟩
        have hwf : u.length < (List.cons a x).length := by
          rw [hs, List.length_cons, List.length_append]
          omega
        /-
          case mp.cons.intro.intro.intro.intro
          α : Type u_1
          inst✝ : DecidableEq α
          P : RegularExpression α
          a : α
          x : List α
          IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
          t u : List α
          hs : Eq x (HAppend.hAppend t u)
          ht : Eq ((P.deriv a).rmatch t) Bool.true
          hu : Eq (P.star.rmatch u) Bool.true
          hwf : LT.lt u.length (List.cons a x).length
          ⊢ Exists fun S => And (Eq (List.cons a x) S.flatten) (∀ (t : List α), Membersh …
        -/
        rw [IH _ hwf] at hu
        /-
          case mp.cons.intro.intro.intro.intro
          α : Type u_1
          inst✝ : DecidableEq α
          P : RegularExpression α
          a : α
          x : List α
          IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
          t u : List α
          hs : Eq x (HAppend.hAppend t u)
          ht : Eq ((P.deriv a).rmatch t) Bool.true
          hu : Exists fun S => And (Eq u S.flatten) (∀ (t : List α), Membership.mem S t  …
          hwf : LT.lt u.length (List.cons a x).length
          ⊢ Exists fun S => And (Eq (List.cons a x) S.flatten) (∀ (t : List α), Membersh …
        -/
        rcases hu with ⟨S', hsum, helem⟩
        /-
          case mp.cons.intro.intro.intro.intro.intro.intro
          α : Type u_1
          inst✝ : DecidableEq α
          P : RegularExpression α
          a : α
          x : List α
          IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
          t u : List α
          hs : Eq x (HAppend.hAppend t u)
          ht : Eq ((P.deriv a).rmatch t) Bool.true
          hwf : LT.lt u.length (List.cons a x).length
          S' : List (List α)
          hsum : Eq u S'.flatten
          helem : ∀ (t : List α), Membership.mem S' t → And (Ne t List.nil) (Eq (P.rmatc …
          ⊢ Exists fun S => And (Eq (List.cons a x) S.flatten) (∀ (t : List α), Membersh …
        -/
        use (a :: t) :: S'
        /-
          case h
          α : Type u_1
          inst✝ : DecidableEq α
          P : RegularExpression α
          a : α
          x : List α
          IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
          t u : List α
          hs : Eq x (HAppend.hAppend t u)
          ht : Eq ((P.deriv a).rmatch t) Bool.true
          hwf : LT.lt u.length (List.cons a x).length
          S' : List (List α)
          hsum : Eq u S'.flatten
          helem : ∀ (t : List α), Membership.mem S' t → And (Ne t List.nil) (Eq (P.rmatc …
          ⊢ And (Eq (List.cons a x) (List.cons (List.cons a t) S').flatten) (∀ (t_1 : Li …
        -/
        constructor
          /-
            case h.left
            α : Type u_1
            inst✝ : DecidableEq α
            P : RegularExpression α
            a : α
            x : List α
            IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
            t u : List α
            hs : Eq x (HAppend.hAppend t u)
            ht : Eq ((P.deriv a).rmatch t) Bool.true
            hwf : LT.lt u.length (List.cons a x).length
            S' : List (List α)
            hsum : Eq u S'.flatten
            helem : ∀ (t : List α), Membership.mem S' t → And (Ne t List.nil) (Eq (P.rmatc …
            ⊢ Eq (List.cons a x) (List.cons (List.cons a t) S').flatten
          -/
        · simp [hs, hsum]
          /-
            🎉 no goals
          -/
          /-
            case h.right
            α : Type u_1
            inst✝ : DecidableEq α
            P : RegularExpression α
            a : α
            x : List α
            IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
            t u : List α
            hs : Eq x (HAppend.hAppend t u)
            ht : Eq ((P.deriv a).rmatch t) Bool.true
            hwf : LT.lt u.length (List.cons a x).length
            S' : List (List α)
            hsum : Eq u S'.flatten
            helem : ∀ (t : List α), Membership.mem S' t → And (Ne t List.nil) (Eq (P.rmatc …
            ⊢ ∀ (t_1 : List α), Membership.mem (List.cons (List.cons a t) S') t_1 → And (N …
          -/
        · intro t' ht'
          cases ht' with
          | head ht' =>
            simp only [ne_eq, not_false_iff, true_and, rmatch, reduceCtorEq]
            exact ht
          | tail _ ht' => exact helem t' ht'
      /-
        case mpr
        α : Type u_1
        inst✝ : DecidableEq α
        P : RegularExpression α
        x : List α
        IH : ∀ (t : List α), LT.lt t.length x.length → Iff (Eq (P.star.rmatch t) Bool. …
        ⊢ (Exists fun S => And (Eq x S.flatten) (∀ (t : List α), Membership.mem S t →  …
      -/
    · rintro ⟨S, hsum, helem⟩
      /-
        case mpr.intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        P : RegularExpression α
        x : List α
        IH : ∀ (t : List α), LT.lt t.length x.length → Iff (Eq (P.star.rmatch t) Bool. …
        S : List (List α)
        hsum : Eq x S.flatten
        helem : ∀ (t : List α), Membership.mem S t → And (Ne t List.nil) (Eq (P.rmatch …
        ⊢ Eq (P.star.rmatch x) Bool.true
      -/
      cases' x with a x
        /-
          case mpr.intro.intro.nil
          α : Type u_1
          inst✝ : DecidableEq α
          P : RegularExpression α
          S : List (List α)
          helem : ∀ (t : List α), Membership.mem S t → And (Ne t List.nil) (Eq (P.rmatch …
          IH : ∀ (t : List α), LT.lt t.length List.nil.length → Iff (Eq (P.star.rmatch t …
          hsum : Eq List.nil S.flatten
          ⊢ Eq (P.star.rmatch List.nil) Bool.true
        -/
      · rfl
        /-
          🎉 no goals
        -/
        /-
          case mpr.intro.intro.cons
          α : Type u_1
          inst✝ : DecidableEq α
          P : RegularExpression α
          S : List (List α)
          helem : ∀ (t : List α), Membership.mem S t → And (Ne t List.nil) (Eq (P.rmatch …
          a : α
          x : List α
          IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
          hsum : Eq (List.cons a x) S.flatten
          ⊢ Eq (P.star.rmatch (List.cons a x)) Bool.true
        -/
      · rw [rmatch, deriv, mul_rmatch_iff]
        /-
          case mpr.intro.intro.cons
          α : Type u_1
          inst✝ : DecidableEq α
          P : RegularExpression α
          S : List (List α)
          helem : ∀ (t : List α), Membership.mem S t → And (Ne t List.nil) (Eq (P.rmatch …
          a : α
          x : List α
          IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
          hsum : Eq (List.cons a x) S.flatten
          ⊢ Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (Eq (( …
        -/
        cases' S with t' U
          /-
            case mpr.intro.intro.cons.nil
            α : Type u_1
            inst✝ : DecidableEq α
            P : RegularExpression α
            a : α
            x : List α
            IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
            helem : ∀ (t : List α), Membership.mem List.nil t → And (Ne t List.nil) (Eq (P …
            hsum : Eq (List.cons a x) List.nil.flatten
            ⊢ Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (Eq (( …
          -/
        · exact ⟨[], [], by tauto⟩
          /-
            🎉 no goals
          -/
          /-
            case mpr.intro.intro.cons.cons
            α : Type u_1
            inst✝ : DecidableEq α
            P : RegularExpression α
            a : α
            x : List α
            IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
            t' : List α
            U : List (List α)
            helem : ∀ (t : List α), Membership.mem (List.cons t' U) t → And (Ne t List.nil …
            hsum : Eq (List.cons a x) (List.cons t' U).flatten
            ⊢ Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (Eq (( …
          -/
        · cases' t' with b t
            /-
              case mpr.intro.intro.cons.cons.nil
              α : Type u_1
              inst✝ : DecidableEq α
              P : RegularExpression α
              a : α
              x : List α
              IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
              U : List (List α)
              helem : ∀ (t : List α), Membership.mem (List.cons List.nil U) t → And (Ne t Li …
              hsum : Eq (List.cons a x) (List.cons List.nil U).flatten
              ⊢ Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (Eq (( …
            -/
          · simp only [forall_eq_or_imp, List.mem_cons] at helem
            /-
              case mpr.intro.intro.cons.cons.nil
              α : Type u_1
              inst✝ : DecidableEq α
              P : RegularExpression α
              a : α
              x : List α
              IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
              U : List (List α)
              hsum : Eq (List.cons a x) (List.cons List.nil U).flatten
              helem : And (And (Ne List.nil List.nil) (Eq (P.rmatch List.nil) Bool.true)) (∀ …
              ⊢ Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (Eq (( …
            -/
            simp only [eq_self_iff_true, not_true, Ne, false_and] at helem
            /-
              🎉 no goals
            -/
          /-
            case mpr.intro.intro.cons.cons.cons
            α : Type u_1
            inst✝ : DecidableEq α
            P : RegularExpression α
            a : α
            x : List α
            IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
            U : List (List α)
            b : α
            t : List α
            helem : ∀ (t_1 : List α), Membership.mem (List.cons (List.cons b t) U) t_1 → A …
            hsum : Eq (List.cons a x) (List.cons (List.cons b t) U).flatten
            ⊢ Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (Eq (( …
          -/
          simp only [List.flatten, List.cons_append, List.cons_eq_cons] at hsum
          /-
            case mpr.intro.intro.cons.cons.cons
            α : Type u_1
            inst✝ : DecidableEq α
            P : RegularExpression α
            a : α
            x : List α
            IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
            U : List (List α)
            b : α
            t : List α
            helem : ∀ (t_1 : List α), Membership.mem (List.cons (List.cons b t) U) t_1 → A …
            hsum : And (Eq a b) (Eq x (HAppend.hAppend t U.flatten))
            ⊢ Exists fun t => Exists fun u => And (Eq x (HAppend.hAppend t u)) (And (Eq (( …
          -/
          refine ⟨t, U.flatten, hsum.2, ?_, ?_⟩
            /-
              case mpr.intro.intro.cons.cons.cons.refine_1
              α : Type u_1
              inst✝ : DecidableEq α
              P : RegularExpression α
              a : α
              x : List α
              IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
              U : List (List α)
              b : α
              t : List α
              helem : ∀ (t_1 : List α), Membership.mem (List.cons (List.cons b t) U) t_1 → A …
              hsum : And (Eq a b) (Eq x (HAppend.hAppend t U.flatten))
              ⊢ Eq ((P.deriv a).rmatch t) Bool.true
            -/
          · specialize helem (b :: t) (by simp)
            /-
              case mpr.intro.intro.cons.cons.cons.refine_1
              α : Type u_1
              inst✝ : DecidableEq α
              P : RegularExpression α
              a : α
              x : List α
              IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
              U : List (List α)
              b : α
              t : List α
              hsum : And (Eq a b) (Eq x (HAppend.hAppend t U.flatten))
              helem : And (Ne (List.cons b t) List.nil) (Eq (P.rmatch (List.cons b t)) Bool. …
              ⊢ Eq ((P.deriv a).rmatch t) Bool.true
            -/
            rw [rmatch] at helem
            /-
              case mpr.intro.intro.cons.cons.cons.refine_1
              α : Type u_1
              inst✝ : DecidableEq α
              P : RegularExpression α
              a : α
              x : List α
              IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
              U : List (List α)
              b : α
              t : List α
              hsum : And (Eq a b) (Eq x (HAppend.hAppend t U.flatten))
              helem : And (Ne (List.cons b t) List.nil) (Eq ((P.deriv b).rmatch t) Bool.true)
              ⊢ Eq ((P.deriv a).rmatch t) Bool.true
            -/
            convert helem.2
            /-
              case h.e'_2.h.e'_3.h.e'_4
              α : Type u_1
              inst✝ : DecidableEq α
              P : RegularExpression α
              a : α
              x : List α
              IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
              U : List (List α)
              b : α
              t : List α
              hsum : And (Eq a b) (Eq x (HAppend.hAppend t U.flatten))
              helem : And (Ne (List.cons b t) List.nil) (Eq ((P.deriv b).rmatch t) Bool.true)
              ⊢ Eq a b
            -/
            exact hsum.1
            /-
              🎉 no goals
            -/
          · have hwf : U.flatten.length < (List.cons a x).length := by
              rw [hsum.1, hsum.2]
              simp only [List.length_append, List.length_flatten, List.length]
              omega
            /-
              case mpr.intro.intro.cons.cons.cons.refine_2
              α : Type u_1
              inst✝ : DecidableEq α
              P : RegularExpression α
              a : α
              x : List α
              IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
              U : List (List α)
              b : α
              t : List α
              helem : ∀ (t_1 : List α), Membership.mem (List.cons (List.cons b t) U) t_1 → A …
              hsum : And (Eq a b) (Eq x (HAppend.hAppend t U.flatten))
              hwf : LT.lt U.flatten.length (List.cons a x).length
              ⊢ Eq (P.star.rmatch U.flatten) Bool.true
            -/
            rw [IH _ hwf]
            /-
              case mpr.intro.intro.cons.cons.cons.refine_2
              α : Type u_1
              inst✝ : DecidableEq α
              P : RegularExpression α
              a : α
              x : List α
              IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
              U : List (List α)
              b : α
              t : List α
              helem : ∀ (t_1 : List α), Membership.mem (List.cons (List.cons b t) U) t_1 → A …
              hsum : And (Eq a b) (Eq x (HAppend.hAppend t U.flatten))
              hwf : LT.lt U.flatten.length (List.cons a x).length
              ⊢ Exists fun S => And (Eq U.flatten S.flatten) (∀ (t : List α), Membership.mem …
            -/
            refine ⟨U, rfl, fun t h => helem t ?_⟩
            /-
              case mpr.intro.intro.cons.cons.cons.refine_2
              α : Type u_1
              inst✝ : DecidableEq α
              P : RegularExpression α
              a : α
              x : List α
              IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
              U : List (List α)
              b : α
              t✝ : List α
              helem : ∀ (t : List α), Membership.mem (List.cons (List.cons b t✝) U) t → And  …
              hsum : And (Eq a b) (Eq x (HAppend.hAppend t✝ U.flatten))
              hwf : LT.lt U.flatten.length (List.cons a x).length
              t : List α
              h : Membership.mem U t
              ⊢ Membership.mem (List.cons (List.cons b t✝) U) t
            -/
            right
            /-
              case mpr.intro.intro.cons.cons.cons.refine_2.a
              α : Type u_1
              inst✝ : DecidableEq α
              P : RegularExpression α
              a : α
              x : List α
              IH : ∀ (t : List α), LT.lt t.length (List.cons a x).length → Iff (Eq (P.star.r …
              U : List (List α)
              b : α
              t✝ : List α
              helem : ∀ (t : List α), Membership.mem (List.cons (List.cons b t✝) U) t → And  …
              hsum : And (Eq a b) (Eq x (HAppend.hAppend t✝ U.flatten))
              hwf : LT.lt U.flatten.length (List.cons a x).length
              t : List α
              h : Membership.mem U t
              ⊢ List.Mem t U
            -/
            assumption
            /-
              🎉 no goals
            -/
  termination_by t => (P, t.length)


@[simp]
theorem rmatch_iff_matches' (P : RegularExpression α) (x : List α) :
    P.rmatch x ↔ x ∈ P.matches' := by
  induction P generalizing x with
  | zero =>
    rw [zero_def, zero_rmatch]
    tauto
  | epsilon =>
    rw [one_def, one_rmatch_iff, matches'_epsilon, Language.mem_one]
  | char =>
    rw [char_rmatch_iff]
    rfl
  | plus _ _ ih₁ ih₂ =>
    rw [plus_def, add_rmatch_iff, ih₁, ih₂]
    rfl
  | comp P Q ih₁ ih₂ =>
    simp only [comp_def, mul_rmatch_iff, matches'_mul, Language.mem_mul, *]
    tauto
  | star _ ih =>
    simp only [star_rmatch_iff, matches'_star, ih, Language.mem_kstar_iff_exists_nonempty, and_comm]


instance (P : RegularExpression α) : DecidablePred (· ∈ P.matches') := fun _ ↦
  decidable_of_iff _ (rmatch_iff_matches' _ _)


/-- Map the alphabet of a regular expression. -/
@[simp]
def map (f : α → β) : RegularExpression α → RegularExpression β
  | 0 => 0
  | 1 => 1
  | char a => char (f a)
  | R + S => map f R + map f S
  | comp R S => map f R * map f S
  | star R => star (map f R)


@[simp]
protected theorem map_pow (f : α → β) (P : RegularExpression α) :
    ∀ n : ℕ, map f (P ^ n) = map f P ^ n
            /-
              α : Type u_1
              β : Type u_2
              f : α → β
              P : RegularExpression α
              ⊢ Eq (RegularExpression.map f (HPow.hPow P 0)) (HPow.hPow (RegularExpression.m …
            -/
  | 0 => by unfold map; rfl
                        /-
                          🎉 no goals
                        -/
  | n + 1 => (congr_arg (· * map f P) (RegularExpression.map_pow f P n) : _)


@[simp]
theorem map_id : ∀ P : RegularExpression α, P.map id = P
  | 0 => rfl
  | 1 => rfl
  | char _ => rfl
                /-
                  α : Type u_1
                  R S : RegularExpression α
                  ⊢ Eq (RegularExpression.map id (HAdd.hAdd R S)) (HAdd.hAdd R S)
                -/
  | R + S => by simp_rw [map, map_id]
                /-
                  🎉 no goals
                -/
                   /-
                     α : Type u_1
                     R S : RegularExpression α
                     ⊢ Eq (RegularExpression.map id (R.comp S)) (R.comp S)
                   -/
  | comp R S => by simp_rw [map, map_id]; rfl
                                          /-
                                            🎉 no goals
                                          -/
                 /-
                   α : Type u_1
                   R : RegularExpression α
                   ⊢ Eq (RegularExpression.map id R.star) R.star
                 -/
  | star R => by simp_rw [map, map_id]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem map_map (g : β → γ) (f : α → β) : ∀ P : RegularExpression α, (P.map f).map g = P.map (g ∘ f)
  | 0 => rfl
  | 1 => rfl
  | char _ => rfl
                /-
                  α : Type u_1
                  β : Type u_2
                  γ : Type u_3
                  g : β → γ
                  f : α → β
                  R S : RegularExpression α
                  ⊢ Eq (RegularExpression.map g (RegularExpression.map f (HAdd.hAdd R S))) (Regu …
                -/
  | R + S => by simp only [map, Function.comp_apply, map_map]
                /-
                  🎉 no goals
                -/
                   /-
                     α : Type u_1
                     β : Type u_2
                     γ : Type u_3
                     g : β → γ
                     f : α → β
                     R S : RegularExpression α
                     ⊢ Eq (RegularExpression.map g (RegularExpression.map f (R.comp S))) (RegularEx …
                   -/
  | comp R S => by simp only [map, Function.comp_apply, map_map]
                   /-
                     🎉 no goals
                   -/
                 /-
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   g : β → γ
                   f : α → β
                   R : RegularExpression α
                   ⊢ Eq (RegularExpression.map g (RegularExpression.map f R.star)) (RegularExpres …
                 -/
  | star R => by simp only [map, Function.comp_apply, map_map]
                 /-
                   🎉 no goals
                 -/


/-- The language of the map is the map of the language. -/
@[simp]
theorem matches'_map (f : α → β) :
    ∀ P : RegularExpression α, (P.map f).matches' = Language.map f P.matches'
  | 0 => (map_zero _).symm
  | 1 => (map_one _).symm
  | char a => by
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      a : α
      ⊢ Eq (RegularExpression.map f (RegularExpression.char a)).matches' ((Language. …
    -/
    rw [eq_comm]
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      a : α
      ⊢ Eq ((Language.map f) (RegularExpression.char a).matches') (RegularExpression …
    -/
    exact image_singleton
    /-
      🎉 no goals
    -/
  -- Porting note: the following close with last `rw` but not with `simp`?
                /-
                  α : Type u_1
                  β : Type u_2
                  f : α → β
                  R S : RegularExpression α
                  ⊢ Eq (RegularExpression.map f (HAdd.hAdd R S)).matches' ((Language.map f) (HAd …
                -/
  | R + S => by simp only [matches'_map, map, matches'_add]; rw [map_add]
                                                             /-
                                                               🎉 no goals
                                                             -/
                   /-
                     α : Type u_1
                     β : Type u_2
                     f : α → β
                     R S : RegularExpression α
                     ⊢ Eq (RegularExpression.map f (R.comp S)).matches' ((Language.map f) (R.comp S …
                   -/
  | comp R S => by simp only [matches'_map, map, matches'_mul]; erw [map_mul]
                                                                /-
                                                                  🎉 no goals
                                                                -/
  | star R => by
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      R : RegularExpression α
      ⊢ Eq (RegularExpression.map f R.star).matches' ((Language.map f) R.star.matche …
    -/
    simp_rw [map, matches', matches'_map]
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      R : RegularExpression α
      ⊢ Eq (KStar.kstar ((Language.map f) R.matches')) ((Language.map f) (KStar.ksta …
    -/
    rw [Language.kstar_eq_iSup_pow, Language.kstar_eq_iSup_pow]
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      R : RegularExpression α
      ⊢ Eq (iSup fun i => HPow.hPow ((Language.map f) R.matches') i) ((Language.map  …
    -/
    simp_rw [← map_pow]
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      R : RegularExpression α
      ⊢ Eq (iSup fun i => (Language.map f) (HPow.hPow R.matches' i)) ((Language.map  …
    -/
    exact image_iUnion.symm
    /-
      🎉 no goals
    -/


