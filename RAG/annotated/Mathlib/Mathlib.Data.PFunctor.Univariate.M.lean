/-- `CofixA F n` is an `n` level approximation of an M-type -/
inductive CofixA : ℕ → Type u
  | continue : CofixA 0
  | intro {n} : ∀ a, (F.B a → CofixA n) → CofixA (succ n)


/-- default inhabitant of `CofixA` -/
protected def CofixA.default [Inhabited F.A] : ∀ n, CofixA F n
  | 0 => CofixA.continue
  | succ n => CofixA.intro default fun _ => CofixA.default n


instance [Inhabited F.A] {n} : Inhabited (CofixA F n) :=
  ⟨CofixA.default F n⟩


theorem cofixA_eq_zero : ∀ x y : CofixA F 0, x = y
  | CofixA.continue, CofixA.continue => rfl


/-- The label of the root of the tree for a non-trivial
approximation of the cofix of a pfunctor.
-/
def head' : ∀ {n}, CofixA F (succ n) → F.A
  | _, CofixA.intro i _ => i


/-- for a non-trivial approximation, return all the subtrees of the root -/
def children' : ∀ {n} (x : CofixA F (succ n)), F.B (head' x) → CofixA F n
  | _, CofixA.intro _ f => f


theorem approx_eta {n : ℕ} (x : CofixA F (n + 1)) : x = CofixA.intro (head' x) (children' x) := by
  /-
    F : PFunctor.{u}
    n : Nat
    x : PFunctor.Approx.CofixA F (HAdd.hAdd n 1)
    ⊢ Eq x (PFunctor.Approx.CofixA.intro (PFunctor.Approx.head' x) (PFunctor.Appro …
  -/
  cases x; rfl
           /-
             🎉 no goals
           -/


/-- Relation between two approximations of the cofix of a pfunctor
that state they both contain the same data until one of them is truncated -/
inductive Agree : ∀ {n : ℕ}, CofixA F n → CofixA F (n + 1) → Prop
  | continu (x : CofixA F 0) (y : CofixA F 1) : Agree x y
  | intro {n} {a} (x : F.B a → CofixA F n) (x' : F.B a → CofixA F (n + 1)) :
    (∀ i : F.B a, Agree (x i) (x' i)) → Agree (CofixA.intro a x) (CofixA.intro a x')


/-- Given an infinite series of approximations `approx`,
`AllAgree approx` states that they are all consistent with each other.
-/
def AllAgree (x : ∀ n, CofixA F n) :=
  ∀ n, Agree (x n) (x (succ n))


@[simp]
                                                                          /-
                                                                            F : PFunctor.{u}
                                                                            x : PFunctor.Approx.CofixA F 0
                                                                            y : PFunctor.Approx.CofixA F 1
                                                                            ⊢ PFunctor.Approx.Agree x y
                                                                          -/
theorem agree_trivial {x : CofixA F 0} {y : CofixA F 1} : Agree x y := by constructor
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[deprecated (since := "2024-12-25")] alias agree_trival := agree_trivial


theorem agree_children {n : ℕ} (x : CofixA F (succ n)) (y : CofixA F (succ n + 1)) {i j}
    (h₀ : HEq i j) (h₁ : Agree x y) : Agree (children' x i) (children' y j) := by
  /-
    F : PFunctor.{u}
    n : Nat
    x : PFunctor.Approx.CofixA F n.succ
    y : PFunctor.Approx.CofixA F (HAdd.hAdd n.succ 1)
    i : F.B (PFunctor.Approx.head' x)
    j : F.B (PFunctor.Approx.head' y)
    h₀ : HEq i j
    h₁ : PFunctor.Approx.Agree x y
    ⊢ PFunctor.Approx.Agree (PFunctor.Approx.children' x i) (PFunctor.Approx.child …
  -/
  cases' h₁ with _ _ _ _ _ _ hagree; cases h₀
  /-
    case intro.refl
    F : PFunctor.{u}
    n : Nat
    a✝ : F.A
    x✝ : F.B a✝ → PFunctor.Approx.CofixA F n
    x'✝ : F.B a✝ → PFunctor.Approx.CofixA F (HAdd.hAdd n 1)
    hagree : ∀ (i : F.B a✝), PFunctor.Approx.Agree (x✝ i) (x'✝ i)
    i : F.B (PFunctor.Approx.head' (PFunctor.Approx.CofixA.intro a✝ x✝))
    ⊢ PFunctor.Approx.Agree (PFunctor.Approx.children' (PFunctor.Approx.CofixA.int …
  -/
  apply hagree
  /-
    🎉 no goals
  -/


/-- `truncate a` turns `a` into a more limited approximation -/
def truncate : ∀ {n : ℕ}, CofixA F (n + 1) → CofixA F n
  | 0, CofixA.intro _ _ => CofixA.continue
  | succ _, CofixA.intro i f => CofixA.intro i <| truncate ∘ f


theorem truncate_eq_of_agree {n : ℕ} (x : CofixA F n) (y : CofixA F (succ n)) (h : Agree x y) :
    truncate y = x := by
  /-
    F : PFunctor.{u}
    n : Nat
    x : PFunctor.Approx.CofixA F n
    y : PFunctor.Approx.CofixA F n.succ
    h : PFunctor.Approx.Agree x y
    ⊢ Eq (PFunctor.Approx.truncate y) x
  -/
  induction n <;> cases x <;> cases y
    /-
      case zero.continue.intro
      F : PFunctor.{u}
      a✝¹ : F.A
      a✝ : F.B a✝¹ → PFunctor.Approx.CofixA F 0
      h : PFunctor.Approx.Agree PFunctor.Approx.CofixA.continue (PFunctor.Approx.Cof …
      ⊢ Eq (PFunctor.Approx.truncate (PFunctor.Approx.CofixA.intro a✝¹ a✝)) PFunctor …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  · -- cases' h with _ _ _ _ _ h₀ h₁
    /-
      case succ.intro.intro
      F : PFunctor.{u}
      n✝ : Nat
      a✝⁴ : ∀ (x : PFunctor.Approx.CofixA F n✝) (y : PFunctor.Approx.CofixA F n✝.suc …
      a✝³ : F.A
      a✝² : F.B a✝³ → PFunctor.Approx.CofixA F n✝
      a✝¹ : F.A
      a✝ : F.B a✝¹ → PFunctor.Approx.CofixA F (HAdd.hAdd n✝ 1)
      h : PFunctor.Approx.Agree (PFunctor.Approx.CofixA.intro a✝³ a✝²) (PFunctor.App …
      ⊢ Eq (PFunctor.Approx.truncate (PFunctor.Approx.CofixA.intro a✝¹ a✝)) (PFuncto …
    -/
    cases h
    /-
      case succ.intro.intro.intro
      F : PFunctor.{u}
      n✝ : Nat
      a✝³ : ∀ (x : PFunctor.Approx.CofixA F n✝) (y : PFunctor.Approx.CofixA F n✝.suc …
      a✝² : F.A
      a✝¹ : F.B a✝² → PFunctor.Approx.CofixA F n✝
      x'✝ : F.B a✝² → PFunctor.Approx.CofixA F (HAdd.hAdd n✝ 1)
      a✝ : ∀ (i : F.B a✝²), PFunctor.Approx.Agree (a✝¹ i) (x'✝ i)
      ⊢ Eq (PFunctor.Approx.truncate (PFunctor.Approx.CofixA.intro a✝² x'✝)) (PFunct …
    -/
    simp only [truncate, Function.comp_def, eq_self_iff_true, heq_iff_eq]
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): used to be `ext y`
    /-
      case succ.intro.intro.intro
      F : PFunctor.{u}
      n✝ : Nat
      a✝³ : ∀ (x : PFunctor.Approx.CofixA F n✝) (y : PFunctor.Approx.CofixA F n✝.suc …
      a✝² : F.A
      a✝¹ : F.B a✝² → PFunctor.Approx.CofixA F n✝
      x'✝ : F.B a✝² → PFunctor.Approx.CofixA F (HAdd.hAdd n✝ 1)
      a✝ : ∀ (i : F.B a✝²), PFunctor.Approx.Agree (a✝¹ i) (x'✝ i)
      ⊢ Eq (PFunctor.Approx.CofixA.intro a✝² fun x => PFunctor.Approx.truncate (x'✝  …
    -/
    rename_i n_ih a f y h₁
    suffices (fun x => truncate (y x)) = f
      by simp [this]
    /-
      case succ.intro.intro.intro
      F : PFunctor.{u}
      n✝ : Nat
      n_ih : ∀ (x : PFunctor.Approx.CofixA F n✝) (y : PFunctor.Approx.CofixA F n✝.su …
      a : F.A
      f : F.B a → PFunctor.Approx.CofixA F n✝
      y : F.B a → PFunctor.Approx.CofixA F (HAdd.hAdd n✝ 1)
      h₁ : ∀ (i : F.B a), PFunctor.Approx.Agree (f i) (y i)
      ⊢ Eq (fun x => PFunctor.Approx.truncate (y x)) f
    -/
    funext y

    /-
      case succ.intro.intro.intro.h
      F : PFunctor.{u}
      n✝ : Nat
      n_ih : ∀ (x : PFunctor.Approx.CofixA F n✝) (y : PFunctor.Approx.CofixA F n✝.su …
      a : F.A
      f : F.B a → PFunctor.Approx.CofixA F n✝
      y✝ : F.B a → PFunctor.Approx.CofixA F (HAdd.hAdd n✝ 1)
      h₁ : ∀ (i : F.B a), PFunctor.Approx.Agree (f i) (y✝ i)
      y : F.B a
      ⊢ Eq (PFunctor.Approx.truncate (y✝ y)) (f y)
    -/
    apply n_ih
    /-
      case succ.intro.intro.intro.h.h
      F : PFunctor.{u}
      n✝ : Nat
      n_ih : ∀ (x : PFunctor.Approx.CofixA F n✝) (y : PFunctor.Approx.CofixA F n✝.su …
      a : F.A
      f : F.B a → PFunctor.Approx.CofixA F n✝
      y✝ : F.B a → PFunctor.Approx.CofixA F (HAdd.hAdd n✝ 1)
      h₁ : ∀ (i : F.B a), PFunctor.Approx.Agree (f i) (y✝ i)
      y : F.B a
      ⊢ PFunctor.Approx.Agree (f y) (y✝ y)
    -/
    apply h₁
    /-
      🎉 no goals
    -/


/-- `sCorec f i n` creates an approximation of height `n`
of the final coalgebra of `f` -/
def sCorec : X → ∀ n, CofixA F n
  | _, 0 => CofixA.continue
  | j, succ _ => CofixA.intro (f j).1 fun i => sCorec ((f j).2 i) _


theorem P_corec (i : X) (n : ℕ) : Agree (sCorec f i n) (sCorec f i (succ n)) := by
  /-
    F : PFunctor.{u}
    X : Type w
    f : X → ↑F X
    i : X
    n : Nat
    ⊢ PFunctor.Approx.Agree (PFunctor.Approx.sCorec f i n) (PFunctor.Approx.sCorec …
  -/
  induction' n with n n_ih generalizing i
  /-
    case zero
    F : PFunctor.{u}
    X : Type w
    f : X → ↑F X
    i : X
    ⊢ PFunctor.Approx.Agree (PFunctor.Approx.sCorec f i 0) (PFunctor.Approx.sCorec …
  -/
  constructor
  /-
    case succ
    F : PFunctor.{u}
    X : Type w
    f : X → ↑F X
    n : Nat
    n_ih : ∀ (i : X), PFunctor.Approx.Agree (PFunctor.Approx.sCorec f i n) (PFunct …
    i : X
    ⊢ PFunctor.Approx.Agree (PFunctor.Approx.sCorec f i (HAdd.hAdd n 1)) (PFunctor …
  -/
  cases' f i with y g
  /-
    case succ.mk
    F : PFunctor.{u}
    X : Type w
    f : X → ↑F X
    n : Nat
    n_ih : ∀ (i : X), PFunctor.Approx.Agree (PFunctor.Approx.sCorec f i n) (PFunct …
    i : X
    y : F.A
    g : F.B y → X
    ⊢ PFunctor.Approx.Agree (PFunctor.Approx.sCorec f i (HAdd.hAdd n 1)) (PFunctor …
  -/
  constructor
  /-
    case succ.mk.a
    F : PFunctor.{u}
    X : Type w
    f : X → ↑F X
    n : Nat
    n_ih : ∀ (i : X), PFunctor.Approx.Agree (PFunctor.Approx.sCorec f i n) (PFunct …
    i : X
    y : F.A
    g : F.B y → X
    ⊢ ∀ (i_1 : F.B (f i).fst), PFunctor.Approx.Agree (PFunctor.Approx.sCorec f ((f …
  -/
  introv
  /-
    case succ.mk.a
    F : PFunctor.{u}
    X : Type w
    f : X → ↑F X
    n : Nat
    n_ih : ∀ (i : X), PFunctor.Approx.Agree (PFunctor.Approx.sCorec f i n) (PFunct …
    i✝ : X
    y : F.A
    g : F.B y → X
    i : F.B (f i✝).fst
    ⊢ PFunctor.Approx.Agree (PFunctor.Approx.sCorec f ((f i✝).snd i) n) (PFunctor. …
  -/
  apply n_ih
  /-
    🎉 no goals
  -/


/-- `Path F` provides indices to access internal nodes in `Corec F` -/
def Path (F : PFunctor.{u}) :=
  List F.Idx


instance Path.inhabited : Inhabited (Path F) :=
  ⟨[]⟩


instance CofixA.instSubsingleton : Subsingleton (CofixA F 0) :=
      /-
        F : PFunctor.{u}
        X : Type w
        f : X → ↑F X
        ⊢ ∀ (a b : PFunctor.Approx.CofixA F 0), Eq a b
      -/
  ⟨by rintro ⟨⟩ ⟨⟩; rfl⟩
                    /-
                      🎉 no goals
                    -/


theorem head_succ' (n m : ℕ) (x : ∀ n, CofixA F n) (Hconsistent : AllAgree x) :
    head' (x (succ n)) = head' (x (succ m)) := by
  /-
    F : PFunctor.{u}
    n m : Nat
    x : (n : Nat) → PFunctor.Approx.CofixA F n
    Hconsistent : PFunctor.Approx.AllAgree x
    ⊢ Eq (PFunctor.Approx.head' (x n.succ)) (PFunctor.Approx.head' (x m.succ))
  -/
  suffices ∀ n, head' (x (succ n)) = head' (x 1) by simp [this]
  /-
    F : PFunctor.{u}
    n m : Nat
    x : (n : Nat) → PFunctor.Approx.CofixA F n
    Hconsistent : PFunctor.Approx.AllAgree x
    ⊢ ∀ (n : Nat), Eq (PFunctor.Approx.head' (x n.succ)) (PFunctor.Approx.head' (x …
  -/
  clear m n
  /-
    F : PFunctor.{u}
    x : (n : Nat) → PFunctor.Approx.CofixA F n
    Hconsistent : PFunctor.Approx.AllAgree x
    ⊢ ∀ (n : Nat), Eq (PFunctor.Approx.head' (x n.succ)) (PFunctor.Approx.head' (x …
  -/
  intro n
  /-
    F : PFunctor.{u}
    x : (n : Nat) → PFunctor.Approx.CofixA F n
    Hconsistent : PFunctor.Approx.AllAgree x
    n : Nat
    ⊢ Eq (PFunctor.Approx.head' (x n.succ)) (PFunctor.Approx.head' (x 1))
  -/
  cases' h₀ : x (succ n) with _ i₀ f₀
  /-
    case intro
    F : PFunctor.{u}
    x : (n : Nat) → PFunctor.Approx.CofixA F n
    Hconsistent : PFunctor.Approx.AllAgree x
    n : Nat
    i₀ : F.A
    f₀ : F.B i₀ → PFunctor.Approx.CofixA F n
    h₀ : Eq (x n.succ) (PFunctor.Approx.CofixA.intro i₀ f₀)
    ⊢ Eq (PFunctor.Approx.head' (PFunctor.Approx.CofixA.intro i₀ f₀)) (PFunctor.Ap …
  -/
  cases' h₁ : x 1 with _ i₁ f₁
  /-
    case intro.intro
    F : PFunctor.{u}
    x : (n : Nat) → PFunctor.Approx.CofixA F n
    Hconsistent : PFunctor.Approx.AllAgree x
    n : Nat
    i₀ : F.A
    f₀ : F.B i₀ → PFunctor.Approx.CofixA F n
    h₀ : Eq (x n.succ) (PFunctor.Approx.CofixA.intro i₀ f₀)
    i₁ : F.A
    f₁ : F.B i₁ → PFunctor.Approx.CofixA F 0
    h₁ : Eq (x 1) (PFunctor.Approx.CofixA.intro i₁ f₁)
    ⊢ Eq (PFunctor.Approx.head' (PFunctor.Approx.CofixA.intro i₀ f₀)) (PFunctor.Ap …
  -/
  dsimp only [head']
  /-
    case intro.intro
    F : PFunctor.{u}
    x : (n : Nat) → PFunctor.Approx.CofixA F n
    Hconsistent : PFunctor.Approx.AllAgree x
    n : Nat
    i₀ : F.A
    f₀ : F.B i₀ → PFunctor.Approx.CofixA F n
    h₀ : Eq (x n.succ) (PFunctor.Approx.CofixA.intro i₀ f₀)
    i₁ : F.A
    f₁ : F.B i₁ → PFunctor.Approx.CofixA F 0
    h₁ : Eq (x 1) (PFunctor.Approx.CofixA.intro i₁ f₁)
    ⊢ Eq (PFunctor.Approx.head'.match_1 (fun x x => F.A) n (PFunctor.Approx.CofixA …
  -/
  induction' n with n n_ih
    /-
      case intro.intro.zero
      F : PFunctor.{u}
      x : (n : Nat) → PFunctor.Approx.CofixA F n
      Hconsistent : PFunctor.Approx.AllAgree x
      i₀ i₁ : F.A
      f₁ : F.B i₁ → PFunctor.Approx.CofixA F 0
      h₁ : Eq (x 1) (PFunctor.Approx.CofixA.intro i₁ f₁)
      f₀ : F.B i₀ → PFunctor.Approx.CofixA F 0
      h₀ : Eq (x (Nat.succ 0)) (PFunctor.Approx.CofixA.intro i₀ f₀)
      ⊢ Eq (PFunctor.Approx.head'.match_1 (fun x x => F.A) 0 (PFunctor.Approx.CofixA …
    -/
  · rw [h₁] at h₀
    /-
      case intro.intro.zero
      F : PFunctor.{u}
      x : (n : Nat) → PFunctor.Approx.CofixA F n
      Hconsistent : PFunctor.Approx.AllAgree x
      i₀ i₁ : F.A
      f₁ : F.B i₁ → PFunctor.Approx.CofixA F 0
      h₁ : Eq (x 1) (PFunctor.Approx.CofixA.intro i₁ f₁)
      f₀ : F.B i₀ → PFunctor.Approx.CofixA F 0
      h₀ : Eq (PFunctor.Approx.CofixA.intro i₁ f₁) (PFunctor.Approx.CofixA.intro i₀  …
      ⊢ Eq (PFunctor.Approx.head'.match_1 (fun x x => F.A) 0 (PFunctor.Approx.CofixA …
    -/
    cases h₀
    /-
      case intro.intro.zero.refl
      F : PFunctor.{u}
      x : (n : Nat) → PFunctor.Approx.CofixA F n
      Hconsistent : PFunctor.Approx.AllAgree x
      i₀ : F.A
      f₀ : F.B i₀ → PFunctor.Approx.CofixA F 0
      h₁ : Eq (x 1) (PFunctor.Approx.CofixA.intro i₀ f₀)
      ⊢ Eq (PFunctor.Approx.head'.match_1 (fun x x => F.A) 0 (PFunctor.Approx.CofixA …
    -/
    trivial
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.succ
      F : PFunctor.{u}
      x : (n : Nat) → PFunctor.Approx.CofixA F n
      Hconsistent : PFunctor.Approx.AllAgree x
      i₀ i₁ : F.A
      f₁ : F.B i₁ → PFunctor.Approx.CofixA F 0
      h₁ : Eq (x 1) (PFunctor.Approx.CofixA.intro i₁ f₁)
      n : Nat
      n_ih : ∀ (f₀ : F.B i₀ → PFunctor.Approx.CofixA F n), Eq (x n.succ) (PFunctor.A …
      f₀ : F.B i₀ → PFunctor.Approx.CofixA F (HAdd.hAdd n 1)
      h₀ : Eq (x (HAdd.hAdd n 1).succ) (PFunctor.Approx.CofixA.intro i₀ f₀)
      ⊢ Eq (PFunctor.Approx.head'.match_1 (fun x x => F.A) (HAdd.hAdd n 1) (PFunctor …
    -/
  · have H := Hconsistent (succ n)
    /-
      case intro.intro.succ
      F : PFunctor.{u}
      x : (n : Nat) → PFunctor.Approx.CofixA F n
      Hconsistent : PFunctor.Approx.AllAgree x
      i₀ i₁ : F.A
      f₁ : F.B i₁ → PFunctor.Approx.CofixA F 0
      h₁ : Eq (x 1) (PFunctor.Approx.CofixA.intro i₁ f₁)
      n : Nat
      n_ih : ∀ (f₀ : F.B i₀ → PFunctor.Approx.CofixA F n), Eq (x n.succ) (PFunctor.A …
      f₀ : F.B i₀ → PFunctor.Approx.CofixA F (HAdd.hAdd n 1)
      h₀ : Eq (x (HAdd.hAdd n 1).succ) (PFunctor.Approx.CofixA.intro i₀ f₀)
      H : PFunctor.Approx.Agree (x n.succ) (x n.succ.succ)
      ⊢ Eq (PFunctor.Approx.head'.match_1 (fun x x => F.A) (HAdd.hAdd n 1) (PFunctor …
    -/
    cases' h₂ : x (succ n) with _ i₂ f₂
    /-
      case intro.intro.succ.intro
      F : PFunctor.{u}
      x : (n : Nat) → PFunctor.Approx.CofixA F n
      Hconsistent : PFunctor.Approx.AllAgree x
      i₀ i₁ : F.A
      f₁ : F.B i₁ → PFunctor.Approx.CofixA F 0
      h₁ : Eq (x 1) (PFunctor.Approx.CofixA.intro i₁ f₁)
      n : Nat
      n_ih : ∀ (f₀ : F.B i₀ → PFunctor.Approx.CofixA F n), Eq (x n.succ) (PFunctor.A …
      f₀ : F.B i₀ → PFunctor.Approx.CofixA F (HAdd.hAdd n 1)
      h₀ : Eq (x (HAdd.hAdd n 1).succ) (PFunctor.Approx.CofixA.intro i₀ f₀)
      H : PFunctor.Approx.Agree (x n.succ) (x n.succ.succ)
      i₂ : F.A
      f₂ : F.B i₂ → PFunctor.Approx.CofixA F n
      h₂ : Eq (x n.succ) (PFunctor.Approx.CofixA.intro i₂ f₂)
      ⊢ Eq (PFunctor.Approx.head'.match_1 (fun x x => F.A) (HAdd.hAdd n 1) (PFunctor …
    -/
    rw [h₀, h₂] at H
    /-
      case intro.intro.succ.intro
      F : PFunctor.{u}
      x : (n : Nat) → PFunctor.Approx.CofixA F n
      Hconsistent : PFunctor.Approx.AllAgree x
      i₀ i₁ : F.A
      f₁ : F.B i₁ → PFunctor.Approx.CofixA F 0
      h₁ : Eq (x 1) (PFunctor.Approx.CofixA.intro i₁ f₁)
      n : Nat
      n_ih : ∀ (f₀ : F.B i₀ → PFunctor.Approx.CofixA F n), Eq (x n.succ) (PFunctor.A …
      f₀ : F.B i₀ → PFunctor.Approx.CofixA F (HAdd.hAdd n 1)
      h₀ : Eq (x (HAdd.hAdd n 1).succ) (PFunctor.Approx.CofixA.intro i₀ f₀)
      i₂ : F.A
      f₂ : F.B i₂ → PFunctor.Approx.CofixA F n
      H : PFunctor.Approx.Agree (PFunctor.Approx.CofixA.intro i₂ f₂) (PFunctor.Appro …
      h₂ : Eq (x n.succ) (PFunctor.Approx.CofixA.intro i₂ f₂)
      ⊢ Eq (PFunctor.Approx.head'.match_1 (fun x x => F.A) (HAdd.hAdd n 1) (PFunctor …
    -/
    apply n_ih (truncate ∘ f₀)
    /-
      case intro.intro.succ.intro
      F : PFunctor.{u}
      x : (n : Nat) → PFunctor.Approx.CofixA F n
      Hconsistent : PFunctor.Approx.AllAgree x
      i₀ i₁ : F.A
      f₁ : F.B i₁ → PFunctor.Approx.CofixA F 0
      h₁ : Eq (x 1) (PFunctor.Approx.CofixA.intro i₁ f₁)
      n : Nat
      n_ih : ∀ (f₀ : F.B i₀ → PFunctor.Approx.CofixA F n), Eq (x n.succ) (PFunctor.A …
      f₀ : F.B i₀ → PFunctor.Approx.CofixA F (HAdd.hAdd n 1)
      h₀ : Eq (x (HAdd.hAdd n 1).succ) (PFunctor.Approx.CofixA.intro i₀ f₀)
      i₂ : F.A
      f₂ : F.B i₂ → PFunctor.Approx.CofixA F n
      H : PFunctor.Approx.Agree (PFunctor.Approx.CofixA.intro i₂ f₂) (PFunctor.Appro …
      h₂ : Eq (x n.succ) (PFunctor.Approx.CofixA.intro i₂ f₂)
      ⊢ Eq (x n.succ) (PFunctor.Approx.CofixA.intro i₀ (Function.comp PFunctor.Appro …
    -/
    rw [h₂]
    /-
      case intro.intro.succ.intro
      F : PFunctor.{u}
      x : (n : Nat) → PFunctor.Approx.CofixA F n
      Hconsistent : PFunctor.Approx.AllAgree x
      i₀ i₁ : F.A
      f₁ : F.B i₁ → PFunctor.Approx.CofixA F 0
      h₁ : Eq (x 1) (PFunctor.Approx.CofixA.intro i₁ f₁)
      n : Nat
      n_ih : ∀ (f₀ : F.B i₀ → PFunctor.Approx.CofixA F n), Eq (x n.succ) (PFunctor.A …
      f₀ : F.B i₀ → PFunctor.Approx.CofixA F (HAdd.hAdd n 1)
      h₀ : Eq (x (HAdd.hAdd n 1).succ) (PFunctor.Approx.CofixA.intro i₀ f₀)
      i₂ : F.A
      f₂ : F.B i₂ → PFunctor.Approx.CofixA F n
      H : PFunctor.Approx.Agree (PFunctor.Approx.CofixA.intro i₂ f₂) (PFunctor.Appro …
      h₂ : Eq (x n.succ) (PFunctor.Approx.CofixA.intro i₂ f₂)
      ⊢ Eq (PFunctor.Approx.CofixA.intro i₂ f₂) (PFunctor.Approx.CofixA.intro i₀ (Fu …
    -/
    cases' H with _ _ _ _ _ _ hagree
    /-
      case intro.intro.succ.intro.intro
      F : PFunctor.{u}
      x : (n : Nat) → PFunctor.Approx.CofixA F n
      Hconsistent : PFunctor.Approx.AllAgree x
      i₀ i₁ : F.A
      f₁ : F.B i₁ → PFunctor.Approx.CofixA F 0
      h₁ : Eq (x 1) (PFunctor.Approx.CofixA.intro i₁ f₁)
      n : Nat
      n_ih : ∀ (f₀ : F.B i₀ → PFunctor.Approx.CofixA F n), Eq (x n.succ) (PFunctor.A …
      f₀ : F.B i₀ → PFunctor.Approx.CofixA F (HAdd.hAdd n 1)
      h₀ : Eq (x (HAdd.hAdd n 1).succ) (PFunctor.Approx.CofixA.intro i₀ f₀)
      f₂ : F.B i₀ → PFunctor.Approx.CofixA F n
      h₂ : Eq (x n.succ) (PFunctor.Approx.CofixA.intro i₀ f₂)
      hagree : ∀ (i : F.B i₀), PFunctor.Approx.Agree (f₂ i) (f₀ i)
      ⊢ Eq (PFunctor.Approx.CofixA.intro i₀ f₂) (PFunctor.Approx.CofixA.intro i₀ (Fu …
    -/
    congr
    /-
      case intro.intro.succ.intro.intro.e_a
      F : PFunctor.{u}
      x : (n : Nat) → PFunctor.Approx.CofixA F n
      Hconsistent : PFunctor.Approx.AllAgree x
      i₀ i₁ : F.A
      f₁ : F.B i₁ → PFunctor.Approx.CofixA F 0
      h₁ : Eq (x 1) (PFunctor.Approx.CofixA.intro i₁ f₁)
      n : Nat
      n_ih : ∀ (f₀ : F.B i₀ → PFunctor.Approx.CofixA F n), Eq (x n.succ) (PFunctor.A …
      f₀ : F.B i₀ → PFunctor.Approx.CofixA F (HAdd.hAdd n 1)
      h₀ : Eq (x (HAdd.hAdd n 1).succ) (PFunctor.Approx.CofixA.intro i₀ f₀)
      f₂ : F.B i₀ → PFunctor.Approx.CofixA F n
      h₂ : Eq (x n.succ) (PFunctor.Approx.CofixA.intro i₀ f₂)
      hagree : ∀ (i : F.B i₀), PFunctor.Approx.Agree (f₂ i) (f₀ i)
      ⊢ Eq f₂ (Function.comp PFunctor.Approx.truncate f₀)
    -/
    funext j
    /-
      case intro.intro.succ.intro.intro.e_a.h
      F : PFunctor.{u}
      x : (n : Nat) → PFunctor.Approx.CofixA F n
      Hconsistent : PFunctor.Approx.AllAgree x
      i₀ i₁ : F.A
      f₁ : F.B i₁ → PFunctor.Approx.CofixA F 0
      h₁ : Eq (x 1) (PFunctor.Approx.CofixA.intro i₁ f₁)
      n : Nat
      n_ih : ∀ (f₀ : F.B i₀ → PFunctor.Approx.CofixA F n), Eq (x n.succ) (PFunctor.A …
      f₀ : F.B i₀ → PFunctor.Approx.CofixA F (HAdd.hAdd n 1)
      h₀ : Eq (x (HAdd.hAdd n 1).succ) (PFunctor.Approx.CofixA.intro i₀ f₀)
      f₂ : F.B i₀ → PFunctor.Approx.CofixA F n
      h₂ : Eq (x n.succ) (PFunctor.Approx.CofixA.intro i₀ f₂)
      hagree : ∀ (i : F.B i₀), PFunctor.Approx.Agree (f₂ i) (f₀ i)
      j : F.B i₀
      ⊢ Eq (f₂ j) (Function.comp PFunctor.Approx.truncate f₀ j)
    -/
    dsimp only [comp_apply]
    /-
      case intro.intro.succ.intro.intro.e_a.h
      F : PFunctor.{u}
      x : (n : Nat) → PFunctor.Approx.CofixA F n
      Hconsistent : PFunctor.Approx.AllAgree x
      i₀ i₁ : F.A
      f₁ : F.B i₁ → PFunctor.Approx.CofixA F 0
      h₁ : Eq (x 1) (PFunctor.Approx.CofixA.intro i₁ f₁)
      n : Nat
      n_ih : ∀ (f₀ : F.B i₀ → PFunctor.Approx.CofixA F n), Eq (x n.succ) (PFunctor.A …
      f₀ : F.B i₀ → PFunctor.Approx.CofixA F (HAdd.hAdd n 1)
      h₀ : Eq (x (HAdd.hAdd n 1).succ) (PFunctor.Approx.CofixA.intro i₀ f₀)
      f₂ : F.B i₀ → PFunctor.Approx.CofixA F n
      h₂ : Eq (x n.succ) (PFunctor.Approx.CofixA.intro i₀ f₂)
      hagree : ∀ (i : F.B i₀), PFunctor.Approx.Agree (f₂ i) (f₀ i)
      j : F.B i₀
      ⊢ Eq (f₂ j) (PFunctor.Approx.truncate (f₀ j))
    -/
    rw [truncate_eq_of_agree]
    /-
      case intro.intro.succ.intro.intro.e_a.h.h
      F : PFunctor.{u}
      x : (n : Nat) → PFunctor.Approx.CofixA F n
      Hconsistent : PFunctor.Approx.AllAgree x
      i₀ i₁ : F.A
      f₁ : F.B i₁ → PFunctor.Approx.CofixA F 0
      h₁ : Eq (x 1) (PFunctor.Approx.CofixA.intro i₁ f₁)
      n : Nat
      n_ih : ∀ (f₀ : F.B i₀ → PFunctor.Approx.CofixA F n), Eq (x n.succ) (PFunctor.A …
      f₀ : F.B i₀ → PFunctor.Approx.CofixA F (HAdd.hAdd n 1)
      h₀ : Eq (x (HAdd.hAdd n 1).succ) (PFunctor.Approx.CofixA.intro i₀ f₀)
      f₂ : F.B i₀ → PFunctor.Approx.CofixA F n
      h₂ : Eq (x n.succ) (PFunctor.Approx.CofixA.intro i₀ f₂)
      hagree : ∀ (i : F.B i₀), PFunctor.Approx.Agree (f₂ i) (f₀ i)
      j : F.B i₀
      ⊢ PFunctor.Approx.Agree (f₂ j) (f₀ j)
    -/
    apply hagree
    /-
      🎉 no goals
    -/


/-- Internal definition for `M`. It is needed to avoid name clashes
between `M.mk` and `M.cases_on` and the declarations generated for
the structure -/
structure MIntl where
  /-- An `n`-th level approximation, for each depth `n` -/
  approx : ∀ n, CofixA F n
  /-- Each approximation agrees with the next -/
  consistent : AllAgree approx


/-- For polynomial functor `F`, `M F` is its final coalgebra -/
def M :=
  MIntl F


theorem M.default_consistent [Inhabited F.A] : ∀ n, Agree (default : CofixA F n) default
  | 0 => Agree.continu _ _
  | succ n => Agree.intro _ _ fun _ => M.default_consistent n


instance M.inhabited [Inhabited F.A] : Inhabited (M F) :=
  ⟨{  approx := default
      consistent := M.default_consistent _ }⟩


instance MIntl.inhabited [Inhabited F.A] : Inhabited (MIntl F) :=
                          /-
                            F : PFunctor.{u}
                            inst✝ : Inhabited F.A
                            ⊢ Inhabited F.M
                          -/
  show Inhabited (M F) by infer_instance
                          /-
                            🎉 no goals
                          -/


theorem ext' (x y : M F) (H : ∀ i : ℕ, x.approx i = y.approx i) : x = y := by
  /-
    F : PFunctor.{u}
    x y : F.M
    H : ∀ (i : Nat), Eq (x.approx i) (y.approx i)
    ⊢ Eq x y
  -/
  cases x
  /-
    case mk
    F : PFunctor.{u}
    y : F.M
    approx✝ : (n : Nat) → PFunctor.Approx.CofixA F n
    consistent✝ : PFunctor.Approx.AllAgree approx✝
    H : ∀ (i : Nat), Eq ({ approx := approx✝, consistent := consistent✝ }.approx i …
    ⊢ Eq { approx := approx✝, consistent := consistent✝ } y
  -/
  cases y
  /-
    case mk.mk
    F : PFunctor.{u}
    approx✝¹ : (n : Nat) → PFunctor.Approx.CofixA F n
    consistent✝¹ : PFunctor.Approx.AllAgree approx✝¹
    approx✝ : (n : Nat) → PFunctor.Approx.CofixA F n
    consistent✝ : PFunctor.Approx.AllAgree approx✝
    H : ∀ (i : Nat), Eq ({ approx := approx✝¹, consistent := consistent✝¹ }.approx …
    ⊢ Eq { approx := approx✝¹, consistent := consistent✝¹ } { approx := approx✝, c …
  -/
  congr with n
  /-
    case mk.mk.e_approx.h
    F : PFunctor.{u}
    approx✝¹ : (n : Nat) → PFunctor.Approx.CofixA F n
    consistent✝¹ : PFunctor.Approx.AllAgree approx✝¹
    approx✝ : (n : Nat) → PFunctor.Approx.CofixA F n
    consistent✝ : PFunctor.Approx.AllAgree approx✝
    H : ∀ (i : Nat), Eq ({ approx := approx✝¹, consistent := consistent✝¹ }.approx …
    n : Nat
    ⊢ Eq (approx✝¹ n) (approx✝ n)
  -/
  apply H
  /-
    🎉 no goals
  -/


/-- Corecursor for the M-type defined by `F`. -/
protected def corec (i : X) : M F where
  approx := sCorec f i
  consistent := P_corec _ _


/-- given a tree generated by `F`, `head` gives us the first piece of data
it contains -/
def head (x : M F) :=
  head' (x.1 1)


/-- return all the subtrees of the root of a tree `x : M F` -/
def children (x : M F) (i : F.B (head x)) : M F :=
  let H := fun n : ℕ => @head_succ' _ n 0 x.1 x.2
                                                                  /-
                                                                    F : PFunctor.{u}
                                                                    X : Type u_1
                                                                    f : X → ↑F X
                                                                    x : F.M
                                                                    i : F.B x.head
                                                                    H : ∀ (n : Nat), Eq (PFunctor.Approx.head' (x.approx n.succ)) (PFunctor.Approx …
                                                                    n : Nat
                                                                    ⊢ Eq x.head (PFunctor.Approx.head' (x.approx n.succ))
                                                                  -/
  { approx := fun n => children' (x.1 _) (cast (congr_arg _ <| by simp only [head, H]) i)
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    consistent := by
      /-
        F : PFunctor.{u}
        X : Type u_1
        f : X → ↑F X
        x : F.M
        i : F.B x.head
        H : ∀ (n : Nat), Eq (PFunctor.Approx.head' (x.approx n.succ)) (PFunctor.Approx …
        ⊢ PFunctor.Approx.AllAgree fun n => PFunctor.Approx.children' (x.approx n.succ …
      -/
      intro n
      /-
        F : PFunctor.{u}
        X : Type u_1
        f : X → ↑F X
        x : F.M
        i : F.B x.head
        H : ∀ (n : Nat), Eq (PFunctor.Approx.head' (x.approx n.succ)) (PFunctor.Approx …
        n : Nat
        ⊢ PFunctor.Approx.Agree ((fun n => PFunctor.Approx.children' (x.approx n.succ) …
      -/
      have P' := x.2 (succ n)
      /-
        F : PFunctor.{u}
        X : Type u_1
        f : X → ↑F X
        x : F.M
        i : F.B x.head
        H : ∀ (n : Nat), Eq (PFunctor.Approx.head' (x.approx n.succ)) (PFunctor.Approx …
        n : Nat
        P' : PFunctor.Approx.Agree (x.approx n.succ) (x.approx n.succ.succ)
        ⊢ PFunctor.Approx.Agree ((fun n => PFunctor.Approx.children' (x.approx n.succ) …
      -/
      apply agree_children _ _ _ P'
      /-
        F : PFunctor.{u}
        X : Type u_1
        f : X → ↑F X
        x : F.M
        i : F.B x.head
        H : ∀ (n : Nat), Eq (PFunctor.Approx.head' (x.approx n.succ)) (PFunctor.Approx …
        n : Nat
        P' : PFunctor.Approx.Agree (x.approx n.succ) (x.approx n.succ.succ)
        ⊢ HEq (cast ⋯ i) (cast ⋯ i)
      -/
      trans i
        /-
          F : PFunctor.{u}
          X : Type u_1
          f : X → ↑F X
          x : F.M
          i : F.B x.head
          H : ∀ (n : Nat), Eq (PFunctor.Approx.head' (x.approx n.succ)) (PFunctor.Approx …
          n : Nat
          P' : PFunctor.Approx.Agree (x.approx n.succ) (x.approx n.succ.succ)
          ⊢ HEq (cast ⋯ i) i
        -/
      · apply cast_heq
        /-
          🎉 no goals
        -/
      /-
        F : PFunctor.{u}
        X : Type u_1
        f : X → ↑F X
        x : F.M
        i : F.B x.head
        H : ∀ (n : Nat), Eq (PFunctor.Approx.head' (x.approx n.succ)) (PFunctor.Approx …
        n : Nat
        P' : PFunctor.Approx.Agree (x.approx n.succ) (x.approx n.succ.succ)
        ⊢ HEq i (cast ⋯ i)
      -/
      symm
      /-
        F : PFunctor.{u}
        X : Type u_1
        f : X → ↑F X
        x : F.M
        i : F.B x.head
        H : ∀ (n : Nat), Eq (PFunctor.Approx.head' (x.approx n.succ)) (PFunctor.Approx …
        n : Nat
        P' : PFunctor.Approx.Agree (x.approx n.succ) (x.approx n.succ.succ)
        ⊢ HEq (cast ⋯ i) i
      -/
      apply cast_heq }
      /-
        🎉 no goals
      -/


/-- select a subtree using an `i : F.Idx` or return an arbitrary tree if
`i` designates no subtree of `x` -/
def ichildren [Inhabited (M F)] [DecidableEq F.A] (i : F.Idx) (x : M F) : M F :=
                                                                /-
                                                                  F : PFunctor.{u}
                                                                  X : Type u_1
                                                                  f : X → ↑F X
                                                                  inst✝¹ : Inhabited F.M
                                                                  inst✝ : DecidableEq F.A
                                                                  i : F.Idx
                                                                  x : F.M
                                                                  H' : Eq i.fst x.head
                                                                  ⊢ Eq i.fst x.head
                                                                -/
  if H' : i.1 = head x then children x (cast (congr_arg _ <| by simp only [head, H']) i.2)
                                                                /-
                                                                  🎉 no goals
                                                                -/
  else default


theorem head_succ (n m : ℕ) (x : M F) : head' (x.approx (succ n)) = head' (x.approx (succ m)) :=
  head_succ' n m _ x.consistent


theorem head_eq_head' : ∀ (x : M F) (n : ℕ), head x = head' (x.approx <| n + 1)
  | ⟨_, h⟩, _ => head_succ' _ _ _ h


theorem head'_eq_head : ∀ (x : M F) (n : ℕ), head' (x.approx <| n + 1) = head x
  | ⟨_, h⟩, _ => head_succ' _ _ _ h


theorem truncate_approx (x : M F) (n : ℕ) : truncate (x.approx <| n + 1) = x.approx n :=
  truncate_eq_of_agree _ _ (x.consistent _)


/-- unfold an M-type -/
def dest : M F → F (M F)
  | x => ⟨head x, fun i => children x i⟩


/-- generates the approximations needed for `M.mk` -/
protected def sMk (x : F (M F)) : ∀ n, CofixA F n
  | 0 => CofixA.continue
  | succ n => CofixA.intro x.1 fun i => (x.2 i).approx n


protected theorem P_mk (x : F (M F)) : AllAgree (Approx.sMk x)
            /-
              F : PFunctor.{u}
              x : ↑F F.M
              ⊢ PFunctor.Approx.Agree (PFunctor.M.Approx.sMk x 0) (PFunctor.M.Approx.sMk x ( …
            -/
  | 0 => by constructor
            /-
              🎉 no goals
            -/
  | succ n => by
    /-
      F : PFunctor.{u}
      x : ↑F F.M
      n : Nat
      ⊢ PFunctor.Approx.Agree (PFunctor.M.Approx.sMk x n.succ) (PFunctor.M.Approx.sM …
    -/
    constructor
    /-
      case a
      F : PFunctor.{u}
      x : ↑F F.M
      n : Nat
      ⊢ ∀ (i : F.B x.fst), PFunctor.Approx.Agree ((x.snd i).approx n) ((x.snd i).app …
    -/
    introv
    /-
      case a
      F : PFunctor.{u}
      x : ↑F F.M
      n : Nat
      i : F.B x.fst
      ⊢ PFunctor.Approx.Agree ((x.snd i).approx n) ((x.snd i).approx (HAdd.hAdd n 1))
    -/
    apply (x.2 i).consistent
    /-
      🎉 no goals
    -/


/-- constructor for M-types -/
protected def mk (x : F (M F)) : M F where
  approx := Approx.sMk x
  consistent := Approx.P_mk x


/-- `Agree' n` relates two trees of type `M F` that
are the same up to depth `n` -/
inductive Agree' : ℕ → M F → M F → Prop
  | trivial (x y : M F) : Agree' 0 x y
  | step {n : ℕ} {a} (x y : F.B a → M F) {x' y'} :
      x' = M.mk ⟨a, x⟩ → y' = M.mk ⟨a, y⟩ → (∀ i, Agree' n (x i) (y i)) → Agree' (succ n) x' y'


@[simp]
theorem dest_mk (x : F (M F)) : dest (M.mk x) = x := rfl


@[simp]
theorem mk_dest (x : M F) : M.mk (dest x) = x := by
  /-
    F : PFunctor.{u}
    x : F.M
    ⊢ Eq (PFunctor.M.mk x.dest) x
  -/
  apply ext'
  /-
    case H
    F : PFunctor.{u}
    x : F.M
    ⊢ ∀ (i : Nat), Eq ((PFunctor.M.mk x.dest).approx i) (x.approx i)
  -/
  intro n
  /-
    case H
    F : PFunctor.{u}
    x : F.M
    n : Nat
    ⊢ Eq ((PFunctor.M.mk x.dest).approx n) (x.approx n)
  -/
  dsimp only [M.mk]
  /-
    case H
    F : PFunctor.{u}
    x : F.M
    n : Nat
    ⊢ Eq (PFunctor.M.Approx.sMk x.dest n) (x.approx n)
  -/
  induction' n with n
    /-
      case H.zero
      F : PFunctor.{u}
      x : F.M
      ⊢ Eq (PFunctor.M.Approx.sMk x.dest 0) (x.approx 0)
    -/
  · apply @Subsingleton.elim _ CofixA.instSubsingleton
    /-
      🎉 no goals
    -/
  /-
    case H.succ
    F : PFunctor.{u}
    x : F.M
    n : Nat
    a✝ : Eq (PFunctor.M.Approx.sMk x.dest n) (x.approx n)
    ⊢ Eq (PFunctor.M.Approx.sMk x.dest (HAdd.hAdd n 1)) (x.approx (HAdd.hAdd n 1))
  -/
  dsimp only [Approx.sMk, dest, head]
  /-
    case H.succ
    F : PFunctor.{u}
    x : F.M
    n : Nat
    a✝ : Eq (PFunctor.M.Approx.sMk x.dest n) (x.approx n)
    ⊢ Eq (PFunctor.Approx.CofixA.intro (PFunctor.Approx.head' (x.approx 1)) fun i  …
  -/
  cases' h : x.approx (succ n) with _ hd ch
  have h' : hd = head' (x.approx 1) := by
    rw [← head_succ' n, h, head']
    apply x.consistent
  /-
    case H.succ.intro
    F : PFunctor.{u}
    x : F.M
    n : Nat
    a✝ : Eq (PFunctor.M.Approx.sMk x.dest n) (x.approx n)
    hd : F.A
    ch : F.B hd → PFunctor.Approx.CofixA F n
    h : Eq (x.approx n.succ) (PFunctor.Approx.CofixA.intro hd ch)
    h' : Eq hd (PFunctor.Approx.head' (x.approx 1))
    ⊢ Eq (PFunctor.Approx.CofixA.intro (PFunctor.Approx.head' (x.approx 1)) fun i  …
  -/
  revert ch
  /-
    case H.succ.intro
    F : PFunctor.{u}
    x : F.M
    n : Nat
    a✝ : Eq (PFunctor.M.Approx.sMk x.dest n) (x.approx n)
    hd : F.A
    h' : Eq hd (PFunctor.Approx.head' (x.approx 1))
    ⊢ ∀ (ch : F.B hd → PFunctor.Approx.CofixA F n), Eq (x.approx n.succ) (PFunctor …
  -/
  rw [h']
  /-
    case H.succ.intro
    F : PFunctor.{u}
    x : F.M
    n : Nat
    a✝ : Eq (PFunctor.M.Approx.sMk x.dest n) (x.approx n)
    hd : F.A
    h' : Eq hd (PFunctor.Approx.head' (x.approx 1))
    ⊢ ∀ (ch : F.B (PFunctor.Approx.head' (x.approx 1)) → PFunctor.Approx.CofixA F  …
  -/
  intros ch h
  /-
    case H.succ.intro
    F : PFunctor.{u}
    x : F.M
    n : Nat
    a✝ : Eq (PFunctor.M.Approx.sMk x.dest n) (x.approx n)
    hd : F.A
    h' : Eq hd (PFunctor.Approx.head' (x.approx 1))
    ch : F.B (PFunctor.Approx.head' (x.approx 1)) → PFunctor.Approx.CofixA F n
    h : Eq (x.approx n.succ) (PFunctor.Approx.CofixA.intro (PFunctor.Approx.head'  …
    ⊢ Eq (PFunctor.Approx.CofixA.intro (PFunctor.Approx.head' (x.approx 1)) fun i  …
  -/
  congr
  /-
    case H.succ.intro.e_a
    F : PFunctor.{u}
    x : F.M
    n : Nat
    a✝ : Eq (PFunctor.M.Approx.sMk x.dest n) (x.approx n)
    hd : F.A
    h' : Eq hd (PFunctor.Approx.head' (x.approx 1))
    ch : F.B (PFunctor.Approx.head' (x.approx 1)) → PFunctor.Approx.CofixA F n
    h : Eq (x.approx n.succ) (PFunctor.Approx.CofixA.intro (PFunctor.Approx.head'  …
    ⊢ Eq (fun i => (x.children i).approx n) ch
  -/
  ext a
  /-
    case H.succ.intro.e_a.h
    F : PFunctor.{u}
    x : F.M
    n : Nat
    a✝ : Eq (PFunctor.M.Approx.sMk x.dest n) (x.approx n)
    hd : F.A
    h' : Eq hd (PFunctor.Approx.head' (x.approx 1))
    ch : F.B (PFunctor.Approx.head' (x.approx 1)) → PFunctor.Approx.CofixA F n
    h : Eq (x.approx n.succ) (PFunctor.Approx.CofixA.intro (PFunctor.Approx.head'  …
    a : F.B (PFunctor.Approx.head' (x.approx 1))
    ⊢ Eq ((x.children a).approx n) (ch a)
  -/
  dsimp only [children]
  /-
    case H.succ.intro.e_a.h
    F : PFunctor.{u}
    x : F.M
    n : Nat
    a✝ : Eq (PFunctor.M.Approx.sMk x.dest n) (x.approx n)
    hd : F.A
    h' : Eq hd (PFunctor.Approx.head' (x.approx 1))
    ch : F.B (PFunctor.Approx.head' (x.approx 1)) → PFunctor.Approx.CofixA F n
    h : Eq (x.approx n.succ) (PFunctor.Approx.CofixA.intro (PFunctor.Approx.head'  …
    a : F.B (PFunctor.Approx.head' (x.approx 1))
    ⊢ Eq (PFunctor.Approx.children' (x.approx n.succ) (cast ⋯ a)) (ch a)
  -/
  generalize hh : cast _ a = a''
  /-
    case H.succ.intro.e_a.h
    F : PFunctor.{u}
    x : F.M
    n : Nat
    a✝ : Eq (PFunctor.M.Approx.sMk x.dest n) (x.approx n)
    hd : F.A
    h' : Eq hd (PFunctor.Approx.head' (x.approx 1))
    ch : F.B (PFunctor.Approx.head' (x.approx 1)) → PFunctor.Approx.CofixA F n
    h : Eq (x.approx n.succ) (PFunctor.Approx.CofixA.intro (PFunctor.Approx.head'  …
    a : F.B (PFunctor.Approx.head' (x.approx 1))
    a'' : F.B (PFunctor.Approx.head' (x.approx n.succ))
    hh : Eq (cast ⋯ a) a''
    ⊢ Eq (PFunctor.Approx.children' (x.approx n.succ) a'') (ch a)
  -/
  rw [cast_eq_iff_heq] at hh
  /-
    case H.succ.intro.e_a.h
    F : PFunctor.{u}
    x : F.M
    n : Nat
    a✝ : Eq (PFunctor.M.Approx.sMk x.dest n) (x.approx n)
    hd : F.A
    h' : Eq hd (PFunctor.Approx.head' (x.approx 1))
    ch : F.B (PFunctor.Approx.head' (x.approx 1)) → PFunctor.Approx.CofixA F n
    h : Eq (x.approx n.succ) (PFunctor.Approx.CofixA.intro (PFunctor.Approx.head'  …
    a : F.B (PFunctor.Approx.head' (x.approx 1))
    a'' : F.B (PFunctor.Approx.head' (x.approx n.succ))
    hh : HEq a a''
    ⊢ Eq (PFunctor.Approx.children' (x.approx n.succ) a'') (ch a)
  -/
  revert a''
  /-
    case H.succ.intro.e_a.h
    F : PFunctor.{u}
    x : F.M
    n : Nat
    a✝ : Eq (PFunctor.M.Approx.sMk x.dest n) (x.approx n)
    hd : F.A
    h' : Eq hd (PFunctor.Approx.head' (x.approx 1))
    ch : F.B (PFunctor.Approx.head' (x.approx 1)) → PFunctor.Approx.CofixA F n
    h : Eq (x.approx n.succ) (PFunctor.Approx.CofixA.intro (PFunctor.Approx.head'  …
    a : F.B (PFunctor.Approx.head' (x.approx 1))
    ⊢ ∀ (a'' : F.B (PFunctor.Approx.head' (x.approx n.succ))), HEq a a'' → Eq (PFu …
  -/
  rw [h]
  /-
    case H.succ.intro.e_a.h
    F : PFunctor.{u}
    x : F.M
    n : Nat
    a✝ : Eq (PFunctor.M.Approx.sMk x.dest n) (x.approx n)
    hd : F.A
    h' : Eq hd (PFunctor.Approx.head' (x.approx 1))
    ch : F.B (PFunctor.Approx.head' (x.approx 1)) → PFunctor.Approx.CofixA F n
    h : Eq (x.approx n.succ) (PFunctor.Approx.CofixA.intro (PFunctor.Approx.head'  …
    a : F.B (PFunctor.Approx.head' (x.approx 1))
    ⊢ ∀ (a'' : F.B (PFunctor.Approx.head' (PFunctor.Approx.CofixA.intro (PFunctor. …
  -/
  intros _ hh
  /-
    case H.succ.intro.e_a.h
    F : PFunctor.{u}
    x : F.M
    n : Nat
    a✝ : Eq (PFunctor.M.Approx.sMk x.dest n) (x.approx n)
    hd : F.A
    h' : Eq hd (PFunctor.Approx.head' (x.approx 1))
    ch : F.B (PFunctor.Approx.head' (x.approx 1)) → PFunctor.Approx.CofixA F n
    h : Eq (x.approx n.succ) (PFunctor.Approx.CofixA.intro (PFunctor.Approx.head'  …
    a : F.B (PFunctor.Approx.head' (x.approx 1))
    a''✝ : F.B (PFunctor.Approx.head' (PFunctor.Approx.CofixA.intro (PFunctor.Appr …
    hh : HEq a a''✝
    ⊢ Eq (PFunctor.Approx.children' (PFunctor.Approx.CofixA.intro (PFunctor.Approx …
  -/
  cases hh
  /-
    case H.succ.intro.e_a.h.refl
    F : PFunctor.{u}
    x : F.M
    n : Nat
    a✝ : Eq (PFunctor.M.Approx.sMk x.dest n) (x.approx n)
    hd : F.A
    h' : Eq hd (PFunctor.Approx.head' (x.approx 1))
    ch : F.B (PFunctor.Approx.head' (x.approx 1)) → PFunctor.Approx.CofixA F n
    h : Eq (x.approx n.succ) (PFunctor.Approx.CofixA.intro (PFunctor.Approx.head'  …
    a : F.B (PFunctor.Approx.head' (x.approx 1))
    ⊢ Eq (PFunctor.Approx.children' (PFunctor.Approx.CofixA.intro (PFunctor.Approx …
  -/
  rfl
  /-
    🎉 no goals
  -/


                                                                   /-
                                                                     F : PFunctor.{u}
                                                                     x y : ↑F F.M
                                                                     h : Eq (PFunctor.M.mk x) (PFunctor.M.mk y)
                                                                     ⊢ Eq x y
                                                                   -/
theorem mk_inj {x y : F (M F)} (h : M.mk x = M.mk y) : x = y := by rw [← dest_mk x, h, dest_mk]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- destructor for M-types -/
protected def cases {r : M F → Sort w} (f : ∀ x : F (M F), r (M.mk x)) (x : M F) : r x :=
  suffices r (M.mk (dest x)) by
    /-
      F : PFunctor.{u}
      X : Type u_1
      f✝ : X → ↑F X
      r : F.M → Sort w
      f : (x : ↑F F.M) → r (PFunctor.M.mk x)
      x : F.M
      this : r (PFunctor.M.mk x.dest)
      ⊢ r x
    -/
    rw [← mk_dest x]
    /-
      F : PFunctor.{u}
      X : Type u_1
      f✝ : X → ↑F X
      r : F.M → Sort w
      f : (x : ↑F F.M) → r (PFunctor.M.mk x)
      x : F.M
      this : r (PFunctor.M.mk x.dest)
      ⊢ r (PFunctor.M.mk x.dest)
    -/
    exact this
    /-
      🎉 no goals
    -/
  f _


/-- destructor for M-types -/
protected def casesOn {r : M F → Sort w} (x : M F) (f : ∀ x : F (M F), r (M.mk x)) : r x :=
  M.cases f x


/-- destructor for M-types, similar to `casesOn` but also
gives access directly to the root and subtrees on an M-type -/
protected def casesOn' {r : M F → Sort w} (x : M F) (f : ∀ a f, r (M.mk ⟨a, f⟩)) : r x :=
  M.casesOn x (fun ⟨a, g⟩ => f a g)


theorem approx_mk (a : F.A) (f : F.B a → M F) (i : ℕ) :
    (M.mk ⟨a, f⟩).approx (succ i) = CofixA.intro a fun j => (f j).approx i :=
  rfl


@[simp]
theorem agree'_refl {n : ℕ} (x : M F) : Agree' n x x := by
  /-
    F : PFunctor.{u}
    n : Nat
    x : F.M
    ⊢ PFunctor.M.Agree' n x x
  -/
  induction' n with _ n_ih generalizing x <;>
  /-
    case zero
    F : PFunctor.{u}
    x : F.M
    ⊢ PFunctor.M.Agree' 0 x x
  -/
                                            /-
                                              🎉 no goals
                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  induction x using PFunctor.M.casesOn' <;> constructor <;> try rfl
  /-
    case succ.f.a
    F : PFunctor.{u}
    n✝ : Nat
    n_ih : ∀ (x : F.M), PFunctor.M.Agree' n✝ x x
    a✝ : F.A
    f✝ : F.B a✝ → F.M
    ⊢ ∀ (i : F.B a✝), PFunctor.M.Agree' n✝ (f✝ i) (f✝ i)
  -/
  intros
  /-
    case succ.f.a
    F : PFunctor.{u}
    n✝ : Nat
    n_ih : ∀ (x : F.M), PFunctor.M.Agree' n✝ x x
    a✝ : F.A
    f✝ : F.B a✝ → F.M
    i✝ : F.B a✝
    ⊢ PFunctor.M.Agree' n✝ (f✝ i✝) (f✝ i✝)
  -/
  apply n_ih
  /-
    🎉 no goals
  -/


theorem agree_iff_agree' {n : ℕ} (x y : M F) :
    Agree (x.approx n) (y.approx <| n + 1) ↔ Agree' n x y := by
  /-
    F : PFunctor.{u}
    n : Nat
    x y : F.M
    ⊢ Iff (PFunctor.Approx.Agree (x.approx n) (y.approx (HAdd.hAdd n 1))) (PFuncto …
  -/
  constructor <;> intro h
    /-
      case mp
      F : PFunctor.{u}
      n : Nat
      x y : F.M
      h : PFunctor.Approx.Agree (x.approx n) (y.approx (HAdd.hAdd n 1))
      ⊢ PFunctor.M.Agree' n x y
    -/
  · induction' n with _ n_ih generalizing x y
      /-
        case mp.zero
        F : PFunctor.{u}
        x y : F.M
        h : PFunctor.Approx.Agree (x.approx 0) (y.approx (HAdd.hAdd 0 1))
        ⊢ PFunctor.M.Agree' 0 x y
      -/
    · constructor
      /-
        🎉 no goals
      -/
      /-
        case mp.succ
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.Approx.Agree (x.approx n✝) (y.approx (HAdd.hAdd …
        x y : F.M
        h : PFunctor.Approx.Agree (x.approx (HAdd.hAdd n✝ 1)) (y.approx (HAdd.hAdd (HA …
        ⊢ PFunctor.M.Agree' (HAdd.hAdd n✝ 1) x y
      -/
    · induction x using PFunctor.M.casesOn'
      /-
        case mp.succ.f
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.Approx.Agree (x.approx n✝) (y.approx (HAdd.hAdd …
        y : F.M
        a✝ : F.A
        f✝ : F.B a✝ → F.M
        h : PFunctor.Approx.Agree ((PFunctor.M.mk ⟨a✝, f✝⟩).approx (HAdd.hAdd n✝ 1)) ( …
        ⊢ PFunctor.M.Agree' (HAdd.hAdd n✝ 1) (PFunctor.M.mk ⟨a✝, f✝⟩) y
      -/
      induction y using PFunctor.M.casesOn'
      /-
        case mp.succ.f.f
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.Approx.Agree (x.approx n✝) (y.approx (HAdd.hAdd …
        a✝¹ : F.A
        f✝¹ : F.B a✝¹ → F.M
        a✝ : F.A
        f✝ : F.B a✝ → F.M
        h : PFunctor.Approx.Agree ((PFunctor.M.mk ⟨a✝¹, f✝¹⟩).approx (HAdd.hAdd n✝ 1)) …
        ⊢ PFunctor.M.Agree' (HAdd.hAdd n✝ 1) (PFunctor.M.mk ⟨a✝¹, f✝¹⟩) (PFunctor.M.mk …
      -/
      simp only [approx_mk] at h
      /-
        case mp.succ.f.f
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.Approx.Agree (x.approx n✝) (y.approx (HAdd.hAdd …
        a✝¹ : F.A
        f✝¹ : F.B a✝¹ → F.M
        a✝ : F.A
        f✝ : F.B a✝ → F.M
        h : PFunctor.Approx.Agree (PFunctor.Approx.CofixA.intro a✝¹ fun j => (f✝¹ j).a …
        ⊢ PFunctor.M.Agree' (HAdd.hAdd n✝ 1) (PFunctor.M.mk ⟨a✝¹, f✝¹⟩) (PFunctor.M.mk …
      -/
      cases' h with _ _ _ _ _ _ hagree
      /-
        case mp.succ.f.f.intro
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.Approx.Agree (x.approx n✝) (y.approx (HAdd.hAdd …
        a✝ : F.A
        f✝¹ f✝ : F.B a✝ → F.M
        hagree : ∀ (i : F.B a✝), PFunctor.Approx.Agree ((fun j => (f✝¹ j).approx n✝) i …
        ⊢ PFunctor.M.Agree' (HAdd.hAdd n✝ 1) (PFunctor.M.mk ⟨a✝, f✝¹⟩) (PFunctor.M.mk  …
      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
      constructor <;> try rfl
      /-
        case mp.succ.f.f.intro.a
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.Approx.Agree (x.approx n✝) (y.approx (HAdd.hAdd …
        a✝ : F.A
        f✝¹ f✝ : F.B a✝ → F.M
        hagree : ∀ (i : F.B a✝), PFunctor.Approx.Agree ((fun j => (f✝¹ j).approx n✝) i …
        ⊢ ∀ (i : F.B a✝), PFunctor.M.Agree' n✝ (f✝¹ i) (f✝ i)
      -/
      intro i
      /-
        case mp.succ.f.f.intro.a
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.Approx.Agree (x.approx n✝) (y.approx (HAdd.hAdd …
        a✝ : F.A
        f✝¹ f✝ : F.B a✝ → F.M
        hagree : ∀ (i : F.B a✝), PFunctor.Approx.Agree ((fun j => (f✝¹ j).approx n✝) i …
        i : F.B a✝
        ⊢ PFunctor.M.Agree' n✝ (f✝¹ i) (f✝ i)
      -/
      apply n_ih
      /-
        case mp.succ.f.f.intro.a.h
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.Approx.Agree (x.approx n✝) (y.approx (HAdd.hAdd …
        a✝ : F.A
        f✝¹ f✝ : F.B a✝ → F.M
        hagree : ∀ (i : F.B a✝), PFunctor.Approx.Agree ((fun j => (f✝¹ j).approx n✝) i …
        i : F.B a✝
        ⊢ PFunctor.Approx.Agree ((f✝¹ i).approx n✝) ((f✝ i).approx (HAdd.hAdd n✝ 1))
      -/
      apply hagree
      /-
        🎉 no goals
      -/
    /-
      case mpr
      F : PFunctor.{u}
      n : Nat
      x y : F.M
      h : PFunctor.M.Agree' n x y
      ⊢ PFunctor.Approx.Agree (x.approx n) (y.approx (HAdd.hAdd n 1))
    -/
  · induction' n with _ n_ih generalizing x y
      /-
        case mpr.zero
        F : PFunctor.{u}
        x y : F.M
        h : PFunctor.M.Agree' 0 x y
        ⊢ PFunctor.Approx.Agree (x.approx 0) (y.approx (HAdd.hAdd 0 1))
      -/
    · constructor
      /-
        🎉 no goals
      -/
      /-
        case mpr.succ
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.M.Agree' n✝ x y → PFunctor.Approx.Agree (x.appr …
        x y : F.M
        h : PFunctor.M.Agree' (HAdd.hAdd n✝ 1) x y
        ⊢ PFunctor.Approx.Agree (x.approx (HAdd.hAdd n✝ 1)) (y.approx (HAdd.hAdd (HAdd …
      -/
    · cases' h with _ _ _ a x' y'
      /-
        case mpr.succ.step
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.M.Agree' n✝ x y → PFunctor.Approx.Agree (x.appr …
        x y : F.M
        a : F.A
        x' y' : F.B a → F.M
        a✝² : ∀ (i : F.B a), PFunctor.M.Agree' n✝ (x' i) (y' i)
        a✝¹ : Eq x (PFunctor.M.mk ⟨a, x'⟩)
        a✝ : Eq y (PFunctor.M.mk ⟨a, y'⟩)
        ⊢ PFunctor.Approx.Agree (x.approx (HAdd.hAdd n✝ 1)) (y.approx (HAdd.hAdd (HAdd …
      -/
      induction' x using PFunctor.M.casesOn' with x_a x_f
      /-
        case mpr.succ.step.f
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.M.Agree' n✝ x y → PFunctor.Approx.Agree (x.appr …
        y : F.M
        a : F.A
        x' y' : F.B a → F.M
        a✝² : ∀ (i : F.B a), PFunctor.M.Agree' n✝ (x' i) (y' i)
        a✝¹ : Eq y (PFunctor.M.mk ⟨a, y'⟩)
        x_a : F.A
        x_f : F.B x_a → F.M
        a✝ : Eq (PFunctor.M.mk ⟨x_a, x_f⟩) (PFunctor.M.mk ⟨a, x'⟩)
        ⊢ PFunctor.Approx.Agree ((PFunctor.M.mk ⟨x_a, x_f⟩).approx (HAdd.hAdd n✝ 1)) ( …
      -/
      induction' y using PFunctor.M.casesOn' with y_a y_f
      /-
        case mpr.succ.step.f.f
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.M.Agree' n✝ x y → PFunctor.Approx.Agree (x.appr …
        a : F.A
        x' y' : F.B a → F.M
        a✝² : ∀ (i : F.B a), PFunctor.M.Agree' n✝ (x' i) (y' i)
        x_a : F.A
        x_f : F.B x_a → F.M
        a✝¹ : Eq (PFunctor.M.mk ⟨x_a, x_f⟩) (PFunctor.M.mk ⟨a, x'⟩)
        y_a : F.A
        y_f : F.B y_a → F.M
        a✝ : Eq (PFunctor.M.mk ⟨y_a, y_f⟩) (PFunctor.M.mk ⟨a, y'⟩)
        ⊢ PFunctor.Approx.Agree ((PFunctor.M.mk ⟨x_a, x_f⟩).approx (HAdd.hAdd n✝ 1)) ( …
      -/
      simp only [approx_mk]
      /-
        case mpr.succ.step.f.f
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.M.Agree' n✝ x y → PFunctor.Approx.Agree (x.appr …
        a : F.A
        x' y' : F.B a → F.M
        a✝² : ∀ (i : F.B a), PFunctor.M.Agree' n✝ (x' i) (y' i)
        x_a : F.A
        x_f : F.B x_a → F.M
        a✝¹ : Eq (PFunctor.M.mk ⟨x_a, x_f⟩) (PFunctor.M.mk ⟨a, x'⟩)
        y_a : F.A
        y_f : F.B y_a → F.M
        a✝ : Eq (PFunctor.M.mk ⟨y_a, y_f⟩) (PFunctor.M.mk ⟨a, y'⟩)
        ⊢ PFunctor.Approx.Agree (PFunctor.Approx.CofixA.intro x_a fun j => (x_f j).app …
      -/
      have h_a_1 := mk_inj ‹M.mk ⟨x_a, x_f⟩ = M.mk ⟨a, x'⟩›
      /-
        case mpr.succ.step.f.f
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.M.Agree' n✝ x y → PFunctor.Approx.Agree (x.appr …
        a : F.A
        x' y' : F.B a → F.M
        a✝² : ∀ (i : F.B a), PFunctor.M.Agree' n✝ (x' i) (y' i)
        x_a : F.A
        x_f : F.B x_a → F.M
        a✝¹ : Eq (PFunctor.M.mk ⟨x_a, x_f⟩) (PFunctor.M.mk ⟨a, x'⟩)
        y_a : F.A
        y_f : F.B y_a → F.M
        a✝ : Eq (PFunctor.M.mk ⟨y_a, y_f⟩) (PFunctor.M.mk ⟨a, y'⟩)
        h_a_1 : Eq ⟨x_a, x_f⟩ ⟨a, x'⟩
        ⊢ PFunctor.Approx.Agree (PFunctor.Approx.CofixA.intro x_a fun j => (x_f j).app …
      -/
      cases h_a_1
      /-
        case mpr.succ.step.f.f.refl
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.M.Agree' n✝ x y → PFunctor.Approx.Agree (x.appr …
        a : F.A
        x' y' : F.B a → F.M
        a✝² : ∀ (i : F.B a), PFunctor.M.Agree' n✝ (x' i) (y' i)
        y_a : F.A
        y_f : F.B y_a → F.M
        a✝¹ : Eq (PFunctor.M.mk ⟨y_a, y_f⟩) (PFunctor.M.mk ⟨a, y'⟩)
        a✝ : Eq (PFunctor.M.mk ⟨a, x'⟩) (PFunctor.M.mk ⟨a, x'⟩)
        ⊢ PFunctor.Approx.Agree (PFunctor.Approx.CofixA.intro a fun j => (x' j).approx …
      -/
      replace h_a_2 := mk_inj ‹M.mk ⟨y_a, y_f⟩ = M.mk ⟨a, y'⟩›
      /-
        case mpr.succ.step.f.f.refl
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.M.Agree' n✝ x y → PFunctor.Approx.Agree (x.appr …
        a : F.A
        x' y' : F.B a → F.M
        a✝² : ∀ (i : F.B a), PFunctor.M.Agree' n✝ (x' i) (y' i)
        y_a : F.A
        y_f : F.B y_a → F.M
        a✝¹ : Eq (PFunctor.M.mk ⟨y_a, y_f⟩) (PFunctor.M.mk ⟨a, y'⟩)
        a✝ : Eq (PFunctor.M.mk ⟨a, x'⟩) (PFunctor.M.mk ⟨a, x'⟩)
        h_a_2 : Eq ⟨y_a, y_f⟩ ⟨a, y'⟩
        ⊢ PFunctor.Approx.Agree (PFunctor.Approx.CofixA.intro a fun j => (x' j).approx …
      -/
      cases h_a_2
      /-
        case mpr.succ.step.f.f.refl.refl
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.M.Agree' n✝ x y → PFunctor.Approx.Agree (x.appr …
        a : F.A
        x' y' : F.B a → F.M
        a✝² : ∀ (i : F.B a), PFunctor.M.Agree' n✝ (x' i) (y' i)
        a✝¹ : Eq (PFunctor.M.mk ⟨a, x'⟩) (PFunctor.M.mk ⟨a, x'⟩)
        a✝ : Eq (PFunctor.M.mk ⟨a, y'⟩) (PFunctor.M.mk ⟨a, y'⟩)
        ⊢ PFunctor.Approx.Agree (PFunctor.Approx.CofixA.intro a fun j => (x' j).approx …
      -/
      constructor
      /-
        case mpr.succ.step.f.f.refl.refl.a
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.M.Agree' n✝ x y → PFunctor.Approx.Agree (x.appr …
        a : F.A
        x' y' : F.B a → F.M
        a✝² : ∀ (i : F.B a), PFunctor.M.Agree' n✝ (x' i) (y' i)
        a✝¹ : Eq (PFunctor.M.mk ⟨a, x'⟩) (PFunctor.M.mk ⟨a, x'⟩)
        a✝ : Eq (PFunctor.M.mk ⟨a, y'⟩) (PFunctor.M.mk ⟨a, y'⟩)
        ⊢ ∀ (i : F.B a), PFunctor.Approx.Agree ((x' i).approx n✝) ((y' i).approx (HAdd …
      -/
      intro i
      /-
        case mpr.succ.step.f.f.refl.refl.a
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.M.Agree' n✝ x y → PFunctor.Approx.Agree (x.appr …
        a : F.A
        x' y' : F.B a → F.M
        a✝² : ∀ (i : F.B a), PFunctor.M.Agree' n✝ (x' i) (y' i)
        a✝¹ : Eq (PFunctor.M.mk ⟨a, x'⟩) (PFunctor.M.mk ⟨a, x'⟩)
        a✝ : Eq (PFunctor.M.mk ⟨a, y'⟩) (PFunctor.M.mk ⟨a, y'⟩)
        i : F.B a
        ⊢ PFunctor.Approx.Agree ((x' i).approx n✝) ((y' i).approx (HAdd.hAdd n✝ 1))
      -/
      apply n_ih
      /-
        case mpr.succ.step.f.f.refl.refl.a.h
        F : PFunctor.{u}
        n✝ : Nat
        n_ih : ∀ (x y : F.M), PFunctor.M.Agree' n✝ x y → PFunctor.Approx.Agree (x.appr …
        a : F.A
        x' y' : F.B a → F.M
        a✝² : ∀ (i : F.B a), PFunctor.M.Agree' n✝ (x' i) (y' i)
        a✝¹ : Eq (PFunctor.M.mk ⟨a, x'⟩) (PFunctor.M.mk ⟨a, x'⟩)
        a✝ : Eq (PFunctor.M.mk ⟨a, y'⟩) (PFunctor.M.mk ⟨a, y'⟩)
        i : F.B a
        ⊢ PFunctor.M.Agree' n✝ (x' i) (y' i)
      -/
      simp [*]
      /-
        🎉 no goals
      -/


@[simp]
theorem cases_mk {r : M F → Sort*} (x : F (M F)) (f : ∀ x : F (M F), r (M.mk x)) :
    PFunctor.M.cases f (M.mk x) = f x := by
  /-
    F : PFunctor.{u}
    r : F.M → Sort u_2
    x : ↑F F.M
    f : (x : ↑F F.M) → r (PFunctor.M.mk x)
    ⊢ Eq (PFunctor.M.cases f (PFunctor.M.mk x)) (f x)
  -/
  dsimp only [M.mk, PFunctor.M.cases, dest, head, Approx.sMk, head']
  /-
    F : PFunctor.{u}
    r : F.M → Sort u_2
    x : ↑F F.M
    f : (x : ↑F F.M) → r (PFunctor.M.mk x)
    ⊢ Eq (⋯.mpr (f ⟨PFunctor.Approx.head'.match_1 (fun x x => F.A) 0 (PFunctor.App …
  -/
  cases x; dsimp only [Approx.sMk]
  /-
    case mk
    F : PFunctor.{u}
    r : F.M → Sort u_2
    f : (x : ↑F F.M) → r (PFunctor.M.mk x)
    fst✝ : F.A
    snd✝ : F.B fst✝ → F.M
    ⊢ Eq (⋯.mpr (f ⟨PFunctor.Approx.head'.match_1 (fun x x => F.A) 0 (PFunctor.App …
  -/
  simp only [Eq.mpr]
  /-
    case mk
    F : PFunctor.{u}
    r : F.M → Sort u_2
    f : (x : ↑F F.M) → r (PFunctor.M.mk x)
    fst✝ : F.A
    snd✝ : F.B fst✝ → F.M
    ⊢ Eq (f ⟨PFunctor.Approx.head'.match_1 (fun x x => F.A) 0 (PFunctor.Approx.Cof …
  -/
  apply congrFun
  /-
    case mk.h
    F : PFunctor.{u}
    r : F.M → Sort u_2
    f : (x : ↑F F.M) → r (PFunctor.M.mk x)
    fst✝ : F.A
    snd✝ : F.B fst✝ → F.M
    ⊢ Eq f f
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem casesOn_mk {r : M F → Sort*} (x : F (M F)) (f : ∀ x : F (M F), r (M.mk x)) :
    PFunctor.M.casesOn (M.mk x) f = f x :=
  cases_mk x f


@[simp]
theorem casesOn_mk' {r : M F → Sort*} {a} (x : F.B a → M F)
    (f : ∀ (a) (f : F.B a → M F), r (M.mk ⟨a, f⟩)) :
    PFunctor.M.casesOn' (M.mk ⟨a, x⟩) f = f a x :=
  @cases_mk F r ⟨a, x⟩ (fun ⟨a, g⟩ => f a g)


/-- `IsPath p x` tells us if `p` is a valid path through `x` -/
inductive IsPath : Path F → M F → Prop
  | nil (x : M F) : IsPath [] x
  |
  cons (xs : Path F) {a} (x : M F) (f : F.B a → M F) (i : F.B a) :
    x = M.mk ⟨a, f⟩ → IsPath xs (f i) → IsPath (⟨a, i⟩ :: xs) x


theorem isPath_cons {xs : Path F} {a a'} {f : F.B a → M F} {i : F.B a'} :
    IsPath (⟨a', i⟩ :: xs) (M.mk ⟨a, f⟩) → a = a' := by
  /-
    F : PFunctor.{u}
    xs : PFunctor.Approx.Path F
    a a' : F.A
    f : F.B a → F.M
    i : F.B a'
    ⊢ PFunctor.M.IsPath (List.cons ⟨a', i⟩ xs) (PFunctor.M.mk ⟨a, f⟩) → Eq a a'
  -/
  generalize h : M.mk ⟨a, f⟩ = x
  /-
    F : PFunctor.{u}
    xs : PFunctor.Approx.Path F
    a a' : F.A
    f : F.B a → F.M
    i : F.B a'
    x : F.M
    h : Eq (PFunctor.M.mk ⟨a, f⟩) x
    ⊢ PFunctor.M.IsPath (List.cons ⟨a', i⟩ xs) x → Eq a a'
  -/
  rintro (_ | ⟨_, _, _, _, rfl, _⟩)
  /-
    case cons
    F : PFunctor.{u}
    xs : PFunctor.Approx.Path F
    a a' : F.A
    f : F.B a → F.M
    i : F.B a'
    f✝ : F.B a' → F.M
    a✝ : PFunctor.M.IsPath xs (f✝ i)
    h : Eq (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a', f✝⟩)
    ⊢ Eq a a'
  -/
  cases mk_inj h
  /-
    case cons.refl
    F : PFunctor.{u}
    xs : PFunctor.Approx.Path F
    a : F.A
    f : F.B a → F.M
    i : F.B a
    a✝ : PFunctor.M.IsPath xs (f i)
    h : Eq (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f⟩)
    ⊢ Eq a a
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem isPath_cons' {xs : Path F} {a} {f : F.B a → M F} {i : F.B a} :
    IsPath (⟨a, i⟩ :: xs) (M.mk ⟨a, f⟩) → IsPath xs (f i) := by
  /-
    F : PFunctor.{u}
    xs : PFunctor.Approx.Path F
    a : F.A
    f : F.B a → F.M
    i : F.B a
    ⊢ PFunctor.M.IsPath (List.cons ⟨a, i⟩ xs) (PFunctor.M.mk ⟨a, f⟩) → PFunctor.M. …
  -/
  generalize h : M.mk ⟨a, f⟩ = x
  /-
    F : PFunctor.{u}
    xs : PFunctor.Approx.Path F
    a : F.A
    f : F.B a → F.M
    i : F.B a
    x : F.M
    h : Eq (PFunctor.M.mk ⟨a, f⟩) x
    ⊢ PFunctor.M.IsPath (List.cons ⟨a, i⟩ xs) x → PFunctor.M.IsPath xs (f i)
  -/
  rintro (_ | ⟨_, _, _, _, rfl, hp⟩)
  /-
    case cons
    F : PFunctor.{u}
    xs : PFunctor.Approx.Path F
    a : F.A
    f : F.B a → F.M
    i : F.B a
    f✝ : F.B a → F.M
    hp : PFunctor.M.IsPath xs (f✝ i)
    h : Eq (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f✝⟩)
    ⊢ PFunctor.M.IsPath xs (f i)
  -/
  cases mk_inj h
  /-
    case cons.refl
    F : PFunctor.{u}
    xs : PFunctor.Approx.Path F
    a : F.A
    f : F.B a → F.M
    i : F.B a
    hp : PFunctor.M.IsPath xs (f i)
    h : Eq (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f⟩)
    ⊢ PFunctor.M.IsPath xs (f i)
  -/
  exact hp
  /-
    🎉 no goals
  -/


/-- follow a path through a value of `M F` and return the subtree
found at the end of the path if it is a valid path for that value and
return a default tree -/
def isubtree [DecidableEq F.A] [Inhabited (M F)] : Path F → M F → M F
  | [], x => x
  | ⟨a, i⟩ :: ps, x =>
    PFunctor.M.casesOn' (r := fun _ => M F) x (fun a' f =>
      if h : a = a' then
                                   /-
                                     F : PFunctor.{u}
                                     X : Type u_1
                                     f✝ : X → ↑F X
                                     inst✝¹ : DecidableEq F.A
                                     inst✝ : Inhabited F.M
                                     a : F.A
                                     i : F.B a
                                     ps : List F.Idx
                                     x : F.M
                                     a' : F.A
                                     f : F.B a' → F.M
                                     h : Eq a a'
                                     ⊢ Eq (F.B a) (F.B a')
                                   -/
        isubtree ps (f <| cast (by rw [h]) i)
                                   /-
                                     🎉 no goals
                                   -/
      else
        default (α := M F)
    )


/-- similar to `isubtree` but returns the data at the end of the path instead
of the whole subtree -/
def iselect [DecidableEq F.A] [Inhabited (M F)] (ps : Path F) : M F → F.A := fun x : M F =>
  head <| isubtree ps x


theorem iselect_eq_default [DecidableEq F.A] [Inhabited (M F)] (ps : Path F) (x : M F)
    (h : ¬IsPath ps x) : iselect ps x = head default := by
  /-
    F : PFunctor.{u}
    inst✝¹ : DecidableEq F.A
    inst✝ : Inhabited F.M
    ps : PFunctor.Approx.Path F
    x : F.M
    h : Not (PFunctor.M.IsPath ps x)
    ⊢ Eq (PFunctor.M.iselect ps x) Inhabited.default.head
  -/
  induction' ps with ps_hd ps_tail ps_ih generalizing x
    /-
      case nil
      F : PFunctor.{u}
      inst✝¹ : DecidableEq F.A
      inst✝ : Inhabited F.M
      x : F.M
      h : Not (PFunctor.M.IsPath List.nil x)
      ⊢ Eq (PFunctor.M.iselect List.nil x) Inhabited.default.head
    -/
  · exfalso
    /-
      case nil
      F : PFunctor.{u}
      inst✝¹ : DecidableEq F.A
      inst✝ : Inhabited F.M
      x : F.M
      h : Not (PFunctor.M.IsPath List.nil x)
      ⊢ False
    -/
    apply h
    /-
      case nil
      F : PFunctor.{u}
      inst✝¹ : DecidableEq F.A
      inst✝ : Inhabited F.M
      x : F.M
      h : Not (PFunctor.M.IsPath List.nil x)
      ⊢ PFunctor.M.IsPath List.nil x
    -/
    constructor
    /-
      🎉 no goals
    -/
    /-
      case cons
      F : PFunctor.{u}
      inst✝¹ : DecidableEq F.A
      inst✝ : Inhabited F.M
      ps_hd : F.Idx
      ps_tail : List F.Idx
      ps_ih : ∀ (x : F.M), Not (PFunctor.M.IsPath ps_tail x) → Eq (PFunctor.M.iselec …
      x : F.M
      h : Not (PFunctor.M.IsPath (List.cons ps_hd ps_tail) x)
      ⊢ Eq (PFunctor.M.iselect (List.cons ps_hd ps_tail) x) Inhabited.default.head
    -/
  · cases' ps_hd with a i
    /-
      case cons.mk
      F : PFunctor.{u}
      inst✝¹ : DecidableEq F.A
      inst✝ : Inhabited F.M
      ps_tail : List F.Idx
      ps_ih : ∀ (x : F.M), Not (PFunctor.M.IsPath ps_tail x) → Eq (PFunctor.M.iselec …
      x : F.M
      a : F.A
      i : F.B a
      h : Not (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps_tail) x)
      ⊢ Eq (PFunctor.M.iselect (List.cons ⟨a, i⟩ ps_tail) x) Inhabited.default.head
    -/
    induction' x using PFunctor.M.casesOn' with x_a x_f
    /-
      case cons.mk.f
      F : PFunctor.{u}
      inst✝¹ : DecidableEq F.A
      inst✝ : Inhabited F.M
      ps_tail : List F.Idx
      ps_ih : ∀ (x : F.M), Not (PFunctor.M.IsPath ps_tail x) → Eq (PFunctor.M.iselec …
      a : F.A
      i : F.B a
      x_a : F.A
      x_f : F.B x_a → F.M
      h : Not (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps_tail) (PFunctor.M.mk ⟨x_a, x_f …
      ⊢ Eq (PFunctor.M.iselect (List.cons ⟨a, i⟩ ps_tail) (PFunctor.M.mk ⟨x_a, x_f⟩) …
    -/
    simp only [iselect, isubtree] at ps_ih ⊢
    /-
      case cons.mk.f
      F : PFunctor.{u}
      inst✝¹ : DecidableEq F.A
      inst✝ : Inhabited F.M
      ps_tail : List F.Idx
      ps_ih : ∀ (x : F.M), Not (PFunctor.M.IsPath ps_tail x) → Eq (PFunctor.M.isubtr …
      a : F.A
      i : F.B a
      x_a : F.A
      x_f : F.B x_a → F.M
      h : Not (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps_tail) (PFunctor.M.mk ⟨x_a, x_f …
      ⊢ Eq ((PFunctor.M.mk ⟨x_a, x_f⟩).casesOn' fun a' f => dite (Eq a a') (fun h => …
    -/
    by_cases h'' : a = x_a
      /-
        case pos
        F : PFunctor.{u}
        inst✝¹ : DecidableEq F.A
        inst✝ : Inhabited F.M
        ps_tail : List F.Idx
        ps_ih : ∀ (x : F.M), Not (PFunctor.M.IsPath ps_tail x) → Eq (PFunctor.M.isubtr …
        a : F.A
        i : F.B a
        x_a : F.A
        x_f : F.B x_a → F.M
        h : Not (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps_tail) (PFunctor.M.mk ⟨x_a, x_f …
        h'' : Eq a x_a
        ⊢ Eq ((PFunctor.M.mk ⟨x_a, x_f⟩).casesOn' fun a' f => dite (Eq a a') (fun h => …
      -/
    · subst x_a
      /-
        case pos
        F : PFunctor.{u}
        inst✝¹ : DecidableEq F.A
        inst✝ : Inhabited F.M
        ps_tail : List F.Idx
        ps_ih : ∀ (x : F.M), Not (PFunctor.M.IsPath ps_tail x) → Eq (PFunctor.M.isubtr …
        a : F.A
        i : F.B a
        x_f : F.B a → F.M
        h : Not (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps_tail) (PFunctor.M.mk ⟨a, x_f⟩))
        ⊢ Eq ((PFunctor.M.mk ⟨a, x_f⟩).casesOn' fun a' f => dite (Eq a a') (fun h => P …
      -/
      simp only [dif_pos, eq_self_iff_true, casesOn_mk']
      /-
        case pos
        F : PFunctor.{u}
        inst✝¹ : DecidableEq F.A
        inst✝ : Inhabited F.M
        ps_tail : List F.Idx
        ps_ih : ∀ (x : F.M), Not (PFunctor.M.IsPath ps_tail x) → Eq (PFunctor.M.isubtr …
        a : F.A
        i : F.B a
        x_f : F.B a → F.M
        h : Not (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps_tail) (PFunctor.M.mk ⟨a, x_f⟩))
        ⊢ Eq (PFunctor.M.isubtree ps_tail (x_f (cast ⋯ i))).head Inhabited.default.head
      -/
      rw [ps_ih]
      /-
        case pos.h
        F : PFunctor.{u}
        inst✝¹ : DecidableEq F.A
        inst✝ : Inhabited F.M
        ps_tail : List F.Idx
        ps_ih : ∀ (x : F.M), Not (PFunctor.M.IsPath ps_tail x) → Eq (PFunctor.M.isubtr …
        a : F.A
        i : F.B a
        x_f : F.B a → F.M
        h : Not (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps_tail) (PFunctor.M.mk ⟨a, x_f⟩))
        ⊢ Not (PFunctor.M.IsPath ps_tail (x_f (cast ⋯ i)))
      -/
      intro h'
      /-
        case pos.h
        F : PFunctor.{u}
        inst✝¹ : DecidableEq F.A
        inst✝ : Inhabited F.M
        ps_tail : List F.Idx
        ps_ih : ∀ (x : F.M), Not (PFunctor.M.IsPath ps_tail x) → Eq (PFunctor.M.isubtr …
        a : F.A
        i : F.B a
        x_f : F.B a → F.M
        h : Not (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps_tail) (PFunctor.M.mk ⟨a, x_f⟩))
        h' : PFunctor.M.IsPath ps_tail (x_f (cast ⋯ i))
        ⊢ False
      -/
      apply h
      /-
        case pos.h
        F : PFunctor.{u}
        inst✝¹ : DecidableEq F.A
        inst✝ : Inhabited F.M
        ps_tail : List F.Idx
        ps_ih : ∀ (x : F.M), Not (PFunctor.M.IsPath ps_tail x) → Eq (PFunctor.M.isubtr …
        a : F.A
        i : F.B a
        x_f : F.B a → F.M
        h : Not (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps_tail) (PFunctor.M.mk ⟨a, x_f⟩))
        h' : PFunctor.M.IsPath ps_tail (x_f (cast ⋯ i))
        ⊢ PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps_tail) (PFunctor.M.mk ⟨a, x_f⟩)
      -/
                      /-
                        🎉 no goals
                      -/
      constructor <;> try rfl
      /-
        case pos.h.a
        F : PFunctor.{u}
        inst✝¹ : DecidableEq F.A
        inst✝ : Inhabited F.M
        ps_tail : List F.Idx
        ps_ih : ∀ (x : F.M), Not (PFunctor.M.IsPath ps_tail x) → Eq (PFunctor.M.isubtr …
        a : F.A
        i : F.B a
        x_f : F.B a → F.M
        h : Not (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps_tail) (PFunctor.M.mk ⟨a, x_f⟩))
        h' : PFunctor.M.IsPath ps_tail (x_f (cast ⋯ i))
        ⊢ PFunctor.M.IsPath ps_tail (x_f i)
      -/
      apply h'
      /-
        🎉 no goals
      -/
      /-
        case neg
        F : PFunctor.{u}
        inst✝¹ : DecidableEq F.A
        inst✝ : Inhabited F.M
        ps_tail : List F.Idx
        ps_ih : ∀ (x : F.M), Not (PFunctor.M.IsPath ps_tail x) → Eq (PFunctor.M.isubtr …
        a : F.A
        i : F.B a
        x_a : F.A
        x_f : F.B x_a → F.M
        h : Not (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps_tail) (PFunctor.M.mk ⟨x_a, x_f …
        h'' : Not (Eq a x_a)
        ⊢ Eq ((PFunctor.M.mk ⟨x_a, x_f⟩).casesOn' fun a' f => dite (Eq a a') (fun h => …
      -/
    · simp [*]
      /-
        🎉 no goals
      -/


@[simp]
theorem head_mk (x : F (M F)) : head (M.mk x) = x.1 :=
  Eq.symm <|
    calc
                                    /-
                                      F : PFunctor.{u}
                                      x : ↑F F.M
                                      ⊢ Eq x.fst (PFunctor.M.mk x).dest.fst
                                    -/
      x.1 = (dest (M.mk x)).1 := by rw [dest_mk]
                                    /-
                                      🎉 no goals
                                    -/
      _ = head (M.mk x) := rfl


theorem children_mk {a} (x : F.B a → M F) (i : F.B (head (M.mk ⟨a, x⟩))) :
                                           /-
                                             F : PFunctor.{u}
                                             X : Type u_1
                                             f : X → ↑F X
                                             a : F.A
                                             x : F.B a → F.M
                                             i : F.B (PFunctor.M.mk ⟨a, x⟩).head
                                             ⊢ Eq (F.B (PFunctor.M.mk ⟨a, x⟩).head) (F.B a)
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
    children (M.mk ⟨a, x⟩) i = x (cast (by rw [head_mk]) i) := by apply ext'; intro n; rfl
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[simp]
theorem ichildren_mk [DecidableEq F.A] [Inhabited (M F)] (x : F (M F)) (i : F.Idx) :
    ichildren i (M.mk x) = x.iget i := by
  /-
    F : PFunctor.{u}
    inst✝¹ : DecidableEq F.A
    inst✝ : Inhabited F.M
    x : ↑F F.M
    i : F.Idx
    ⊢ Eq (PFunctor.M.ichildren i (PFunctor.M.mk x)) (x.iget i)
  -/
  dsimp only [ichildren, PFunctor.Obj.iget]
  /-
    F : PFunctor.{u}
    inst✝¹ : DecidableEq F.A
    inst✝ : Inhabited F.M
    x : ↑F F.M
    i : F.Idx
    ⊢ Eq (dite (Eq i.fst (PFunctor.M.mk x).head) (fun H' => (PFunctor.M.mk x).chil …
  -/
  congr with h
  /-
    🎉 no goals
  -/


@[simp]
theorem isubtree_cons [DecidableEq F.A] [Inhabited (M F)] (ps : Path F) {a} (f : F.B a → M F)
    {i : F.B a} : isubtree (⟨_, i⟩ :: ps) (M.mk ⟨a, f⟩) = isubtree ps (f i) := by
  /-
    F : PFunctor.{u}
    inst✝¹ : DecidableEq F.A
    inst✝ : Inhabited F.M
    ps : PFunctor.Approx.Path F
    a : F.A
    f : F.B a → F.M
    i : F.B a
    ⊢ Eq (PFunctor.M.isubtree (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)) (PFunc …
  -/
  simp only [isubtree, ichildren_mk, PFunctor.Obj.iget, dif_pos, isubtree, M.casesOn_mk']; rfl
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


@[simp]
theorem iselect_nil [DecidableEq F.A] [Inhabited (M F)] {a} (f : F.B a → M F) :
    iselect nil (M.mk ⟨a, f⟩) = a := rfl


@[simp]
theorem iselect_cons [DecidableEq F.A] [Inhabited (M F)] (ps : Path F) {a} (f : F.B a → M F) {i} :
                                                                  /-
                                                                    F : PFunctor.{u}
                                                                    inst✝¹ : DecidableEq F.A
                                                                    inst✝ : Inhabited F.M
                                                                    ps : PFunctor.Approx.Path F
                                                                    a : F.A
                                                                    f : F.B a → F.M
                                                                    i : F.B a
                                                                    ⊢ Eq (PFunctor.M.iselect (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)) (PFunct …
                                                                  -/
    iselect (⟨a, i⟩ :: ps) (M.mk ⟨a, f⟩) = iselect ps (f i) := by simp only [iselect, isubtree_cons]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem corec_def {X} (f : X → F X) (x₀ : X) : M.corec f x₀ = M.mk (F.map (M.corec f) (f x₀)) := by
  /-
    F : PFunctor.{u}
    X : Type u_2
    f : X → ↑F X
    x₀ : X
    ⊢ Eq (PFunctor.M.corec f x₀) (PFunctor.M.mk (F.map (PFunctor.M.corec f) (f x₀)))
  -/
  dsimp only [M.corec, M.mk]
  /-
    F : PFunctor.{u}
    X : Type u_2
    f : X → ↑F X
    x₀ : X
    ⊢ Eq { approx := PFunctor.Approx.sCorec f x₀, consistent := ⋯ } { approx := PF …
  -/
  congr with n
  /-
    case e_approx.h
    F : PFunctor.{u}
    X : Type u_2
    f : X → ↑F X
    x₀ : X
    n : Nat
    ⊢ Eq (PFunctor.Approx.sCorec f x₀ n) (PFunctor.M.Approx.sMk (F.map (PFunctor.M …
  -/
  cases' n with n
    /-
      case e_approx.h.zero
      F : PFunctor.{u}
      X : Type u_2
      f : X → ↑F X
      x₀ : X
      ⊢ Eq (PFunctor.Approx.sCorec f x₀ 0) (PFunctor.M.Approx.sMk (F.map (PFunctor.M …
    -/
  · dsimp only [sCorec, Approx.sMk]
    /-
      🎉 no goals
    -/
    /-
      case e_approx.h.succ
      F : PFunctor.{u}
      X : Type u_2
      f : X → ↑F X
      x₀ : X
      n : Nat
      ⊢ Eq (PFunctor.Approx.sCorec f x₀ (HAdd.hAdd n 1)) (PFunctor.M.Approx.sMk (F.m …
    -/
  · dsimp only [sCorec, Approx.sMk]
    /-
      case e_approx.h.succ
      F : PFunctor.{u}
      X : Type u_2
      f : X → ↑F X
      x₀ : X
      n : Nat
      ⊢ Eq (PFunctor.Approx.CofixA.intro (f x₀).fst fun i => PFunctor.Approx.sCorec  …
    -/
    cases f x₀
    /-
      case e_approx.h.succ.mk
      F : PFunctor.{u}
      X : Type u_2
      f : X → ↑F X
      x₀ : X
      n : Nat
      fst✝ : F.A
      snd✝ : F.B fst✝ → X
      ⊢ Eq (PFunctor.Approx.CofixA.intro ⟨fst✝, snd✝⟩.fst fun i => PFunctor.Approx.s …
    -/
    dsimp only [PFunctor.map]
    /-
      case e_approx.h.succ.mk
      F : PFunctor.{u}
      X : Type u_2
      f : X → ↑F X
      x₀ : X
      n : Nat
      fst✝ : F.A
      snd✝ : F.B fst✝ → X
      ⊢ Eq (PFunctor.Approx.CofixA.intro fst✝ fun i => PFunctor.Approx.sCorec f (snd …
    -/
    congr
    /-
      🎉 no goals
    -/


theorem ext_aux [Inhabited (M F)] [DecidableEq F.A] {n : ℕ} (x y z : M F) (hx : Agree' n z x)
    (hy : Agree' n z y) (hrec : ∀ ps : Path F, n = ps.length → iselect ps x = iselect ps y) :
    x.approx (n + 1) = y.approx (n + 1) := by
  /-
    F : PFunctor.{u}
    inst✝¹ : Inhabited F.M
    inst✝ : DecidableEq F.A
    n : Nat
    x y z : F.M
    hx : PFunctor.M.Agree' n z x
    hy : PFunctor.M.Agree' n z y
    hrec : ∀ (ps : PFunctor.Approx.Path F), Eq n (List.length ps) → Eq (PFunctor.M …
    ⊢ Eq (x.approx (HAdd.hAdd n 1)) (y.approx (HAdd.hAdd n 1))
  -/
  induction' n with n n_ih generalizing x y z
    /-
      case zero
      F : PFunctor.{u}
      inst✝¹ : Inhabited F.M
      inst✝ : DecidableEq F.A
      x y z : F.M
      hx : PFunctor.M.Agree' 0 z x
      hy : PFunctor.M.Agree' 0 z y
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq 0 (List.length ps) → Eq (PFunctor.M …
      ⊢ Eq (x.approx (HAdd.hAdd 0 1)) (y.approx (HAdd.hAdd 0 1))
    -/
  · specialize hrec [] rfl
    /-
      case zero
      F : PFunctor.{u}
      inst✝¹ : Inhabited F.M
      inst✝ : DecidableEq F.A
      x y z : F.M
      hx : PFunctor.M.Agree' 0 z x
      hy : PFunctor.M.Agree' 0 z y
      hrec : Eq (PFunctor.M.iselect List.nil x) (PFunctor.M.iselect List.nil y)
      ⊢ Eq (x.approx (HAdd.hAdd 0 1)) (y.approx (HAdd.hAdd 0 1))
    -/
    induction x using PFunctor.M.casesOn'
    /-
      case zero.f
      F : PFunctor.{u}
      inst✝¹ : Inhabited F.M
      inst✝ : DecidableEq F.A
      y z : F.M
      hy : PFunctor.M.Agree' 0 z y
      a✝ : F.A
      f✝ : F.B a✝ → F.M
      hx : PFunctor.M.Agree' 0 z (PFunctor.M.mk ⟨a✝, f✝⟩)
      hrec : Eq (PFunctor.M.iselect List.nil (PFunctor.M.mk ⟨a✝, f✝⟩)) (PFunctor.M.i …
      ⊢ Eq ((PFunctor.M.mk ⟨a✝, f✝⟩).approx (HAdd.hAdd 0 1)) (y.approx (HAdd.hAdd 0  …
    -/
    induction y using PFunctor.M.casesOn'
    /-
      case zero.f.f
      F : PFunctor.{u}
      inst✝¹ : Inhabited F.M
      inst✝ : DecidableEq F.A
      z : F.M
      a✝¹ : F.A
      f✝¹ : F.B a✝¹ → F.M
      hx : PFunctor.M.Agree' 0 z (PFunctor.M.mk ⟨a✝¹, f✝¹⟩)
      a✝ : F.A
      f✝ : F.B a✝ → F.M
      hy : PFunctor.M.Agree' 0 z (PFunctor.M.mk ⟨a✝, f✝⟩)
      hrec : Eq (PFunctor.M.iselect List.nil (PFunctor.M.mk ⟨a✝¹, f✝¹⟩)) (PFunctor.M …
      ⊢ Eq ((PFunctor.M.mk ⟨a✝¹, f✝¹⟩).approx (HAdd.hAdd 0 1)) ((PFunctor.M.mk ⟨a✝,  …
    -/
    simp only [iselect_nil] at hrec
    /-
      case zero.f.f
      F : PFunctor.{u}
      inst✝¹ : Inhabited F.M
      inst✝ : DecidableEq F.A
      z : F.M
      a✝¹ : F.A
      f✝¹ : F.B a✝¹ → F.M
      hx : PFunctor.M.Agree' 0 z (PFunctor.M.mk ⟨a✝¹, f✝¹⟩)
      a✝ : F.A
      f✝ : F.B a✝ → F.M
      hy : PFunctor.M.Agree' 0 z (PFunctor.M.mk ⟨a✝, f✝⟩)
      hrec : Eq a✝¹ a✝
      ⊢ Eq ((PFunctor.M.mk ⟨a✝¹, f✝¹⟩).approx (HAdd.hAdd 0 1)) ((PFunctor.M.mk ⟨a✝,  …
    -/
    subst hrec
    simp only [approx_mk, eq_self_iff_true, heq_iff_eq, zero_eq, CofixA.intro.injEq,
      heq_eq_eq, eq_iff_true_of_subsingleton, and_self]
    /-
      case succ
      F : PFunctor.{u}
      inst✝¹ : Inhabited F.M
      inst✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      x y z : F.M
      hx : PFunctor.M.Agree' (HAdd.hAdd n 1) z x
      hy : PFunctor.M.Agree' (HAdd.hAdd n 1) z y
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      ⊢ Eq (x.approx (HAdd.hAdd (HAdd.hAdd n 1) 1)) (y.approx (HAdd.hAdd (HAdd.hAdd  …
    -/
  · cases hx
    /-
      case succ.step
      F : PFunctor.{u}
      inst✝¹ : Inhabited F.M
      inst✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      x y z : F.M
      hy : PFunctor.M.Agree' (HAdd.hAdd n 1) z y
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      a✝³ : F.A
      x✝ y✝ : F.B a✝³ → F.M
      a✝² : ∀ (i : F.B a✝³), PFunctor.M.Agree' n (x✝ i) (y✝ i)
      a✝¹ : Eq z (PFunctor.M.mk ⟨a✝³, x✝⟩)
      a✝ : Eq x (PFunctor.M.mk ⟨a✝³, y✝⟩)
      ⊢ Eq (x.approx (HAdd.hAdd (HAdd.hAdd n 1) 1)) (y.approx (HAdd.hAdd (HAdd.hAdd  …
    -/
    cases hy
    /-
      case succ.step.step
      F : PFunctor.{u}
      inst✝¹ : Inhabited F.M
      inst✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      x y z : F.M
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      a✝⁷ : F.A
      x✝¹ y✝¹ : F.B a✝⁷ → F.M
      a✝⁶ : ∀ (i : F.B a✝⁷), PFunctor.M.Agree' n (x✝¹ i) (y✝¹ i)
      a✝⁵ : Eq z (PFunctor.M.mk ⟨a✝⁷, x✝¹⟩)
      a✝⁴ : Eq x (PFunctor.M.mk ⟨a✝⁷, y✝¹⟩)
      a✝³ : F.A
      x✝ y✝ : F.B a✝³ → F.M
      a✝² : ∀ (i : F.B a✝³), PFunctor.M.Agree' n (x✝ i) (y✝ i)
      a✝¹ : Eq z (PFunctor.M.mk ⟨a✝³, x✝⟩)
      a✝ : Eq y (PFunctor.M.mk ⟨a✝³, y✝⟩)
      ⊢ Eq (x.approx (HAdd.hAdd (HAdd.hAdd n 1) 1)) (y.approx (HAdd.hAdd (HAdd.hAdd  …
    -/
    induction x using PFunctor.M.casesOn'
    /-
      case succ.step.step.f
      F : PFunctor.{u}
      inst✝¹ : Inhabited F.M
      inst✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      y z : F.M
      a✝⁸ : F.A
      x✝¹ y✝¹ : F.B a✝⁸ → F.M
      a✝⁷ : ∀ (i : F.B a✝⁸), PFunctor.M.Agree' n (x✝¹ i) (y✝¹ i)
      a✝⁶ : Eq z (PFunctor.M.mk ⟨a✝⁸, x✝¹⟩)
      a✝⁵ : F.A
      x✝ y✝ : F.B a✝⁵ → F.M
      a✝⁴ : ∀ (i : F.B a✝⁵), PFunctor.M.Agree' n (x✝ i) (y✝ i)
      a✝³ : Eq z (PFunctor.M.mk ⟨a✝⁵, x✝⟩)
      a✝² : Eq y (PFunctor.M.mk ⟨a✝⁵, y✝⟩)
      a✝¹ : F.A
      f✝ : F.B a✝¹ → F.M
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      a✝ : Eq (PFunctor.M.mk ⟨a✝¹, f✝⟩) (PFunctor.M.mk ⟨a✝⁸, y✝¹⟩)
      ⊢ Eq ((PFunctor.M.mk ⟨a✝¹, f✝⟩).approx (HAdd.hAdd (HAdd.hAdd n 1) 1)) (y.appro …
    -/
    induction y using PFunctor.M.casesOn'
    /-
      case succ.step.step.f.f
      F : PFunctor.{u}
      inst✝¹ : Inhabited F.M
      inst✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      z : F.M
      a✝⁹ : F.A
      x✝¹ y✝¹ : F.B a✝⁹ → F.M
      a✝⁸ : ∀ (i : F.B a✝⁹), PFunctor.M.Agree' n (x✝¹ i) (y✝¹ i)
      a✝⁷ : Eq z (PFunctor.M.mk ⟨a✝⁹, x✝¹⟩)
      a✝⁶ : F.A
      x✝ y✝ : F.B a✝⁶ → F.M
      a✝⁵ : ∀ (i : F.B a✝⁶), PFunctor.M.Agree' n (x✝ i) (y✝ i)
      a✝⁴ : Eq z (PFunctor.M.mk ⟨a✝⁶, x✝⟩)
      a✝³ : F.A
      f✝¹ : F.B a✝³ → F.M
      a✝² : Eq (PFunctor.M.mk ⟨a✝³, f✝¹⟩) (PFunctor.M.mk ⟨a✝⁹, y✝¹⟩)
      a✝¹ : F.A
      f✝ : F.B a✝¹ → F.M
      a✝ : Eq (PFunctor.M.mk ⟨a✝¹, f✝⟩) (PFunctor.M.mk ⟨a✝⁶, y✝⟩)
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      ⊢ Eq ((PFunctor.M.mk ⟨a✝³, f✝¹⟩).approx (HAdd.hAdd (HAdd.hAdd n 1) 1)) ((PFunc …
    -/
    subst z
    /-
      case succ.step.step.f.f
      F : PFunctor.{u}
      inst✝¹ : Inhabited F.M
      inst✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      a✝⁸ : F.A
      x✝¹ y✝¹ : F.B a✝⁸ → F.M
      a✝⁷ : ∀ (i : F.B a✝⁸), PFunctor.M.Agree' n (x✝¹ i) (y✝¹ i)
      a✝⁶ : F.A
      x✝ y✝ : F.B a✝⁶ → F.M
      a✝⁵ : ∀ (i : F.B a✝⁶), PFunctor.M.Agree' n (x✝ i) (y✝ i)
      a✝⁴ : F.A
      f✝¹ : F.B a✝⁴ → F.M
      a✝³ : Eq (PFunctor.M.mk ⟨a✝⁴, f✝¹⟩) (PFunctor.M.mk ⟨a✝⁸, y✝¹⟩)
      a✝² : F.A
      f✝ : F.B a✝² → F.M
      a✝¹ : Eq (PFunctor.M.mk ⟨a✝², f✝⟩) (PFunctor.M.mk ⟨a✝⁶, y✝⟩)
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      a✝ : Eq (PFunctor.M.mk ⟨a✝⁸, x✝¹⟩) (PFunctor.M.mk ⟨a✝⁶, x✝⟩)
      ⊢ Eq ((PFunctor.M.mk ⟨a✝⁴, f✝¹⟩).approx (HAdd.hAdd (HAdd.hAdd n 1) 1)) ((PFunc …
    -/
    iterate 3 (have := mk_inj ‹_›; cases this)
    /-
      case succ.step.step.f.f.refl.refl.refl
      F : PFunctor.{u}
      inst✝¹ : Inhabited F.M
      inst✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      a✝⁷ : F.A
      x✝ y✝¹ : F.B a✝⁷ → F.M
      a✝⁶ : ∀ (i : F.B a✝⁷), PFunctor.M.Agree' n (x✝ i) (y✝¹ i)
      a✝⁵ : F.A
      f✝¹ : F.B a✝⁵ → F.M
      a✝⁴ : Eq (PFunctor.M.mk ⟨a✝⁵, f✝¹⟩) (PFunctor.M.mk ⟨a✝⁷, y✝¹⟩)
      a✝³ : F.A
      f✝ : F.B a✝³ → F.M
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      y✝ : F.B a✝⁷ → F.M
      a✝² : Eq (PFunctor.M.mk ⟨a✝³, f✝⟩) (PFunctor.M.mk ⟨a✝⁷, y✝⟩)
      a✝¹ : ∀ (i : F.B a✝⁷), PFunctor.M.Agree' n (x✝ i) (y✝ i)
      a✝ : Eq (PFunctor.M.mk ⟨a✝⁷, x✝⟩) (PFunctor.M.mk ⟨a✝⁷, x✝⟩)
      ⊢ Eq ((PFunctor.M.mk ⟨a✝⁵, f✝¹⟩).approx (HAdd.hAdd (HAdd.hAdd n 1) 1)) ((PFunc …
    -/
    rename_i n_ih a f₃ f₂ hAgree₂ _ _ h₂ _ _ f₁ h₁ hAgree₁ clr
    /-
      case succ.step.step.f.f.refl.refl.refl
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      n_ih✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      a : F.A
      f₃ f₂ : F.B a → F.M
      hAgree₂ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₂ i)
      a✝¹ : F.A
      f✝¹ : F.B a✝¹ → F.M
      h₂ : Eq (PFunctor.M.mk ⟨a✝¹, f✝¹⟩) (PFunctor.M.mk ⟨a, f₂⟩)
      a✝ : F.A
      f✝ : F.B a✝ → F.M
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      f₁ : F.B a → F.M
      h₁ : Eq (PFunctor.M.mk ⟨a✝, f✝⟩) (PFunctor.M.mk ⟨a, f₁⟩)
      hAgree₁ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₁ i)
      clr : Eq (PFunctor.M.mk ⟨a, f₃⟩) (PFunctor.M.mk ⟨a, f₃⟩)
      ⊢ Eq ((PFunctor.M.mk ⟨a✝¹, f✝¹⟩).approx (HAdd.hAdd (HAdd.hAdd n 1) 1)) ((PFunc …
    -/
    simp only [approx_mk, eq_self_iff_true, heq_iff_eq]

    /-
      case succ.step.step.f.f.refl.refl.refl
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      n_ih✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      a : F.A
      f₃ f₂ : F.B a → F.M
      hAgree₂ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₂ i)
      a✝¹ : F.A
      f✝¹ : F.B a✝¹ → F.M
      h₂ : Eq (PFunctor.M.mk ⟨a✝¹, f✝¹⟩) (PFunctor.M.mk ⟨a, f₂⟩)
      a✝ : F.A
      f✝ : F.B a✝ → F.M
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      f₁ : F.B a → F.M
      h₁ : Eq (PFunctor.M.mk ⟨a✝, f✝⟩) (PFunctor.M.mk ⟨a, f₁⟩)
      hAgree₁ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₁ i)
      clr : Eq (PFunctor.M.mk ⟨a, f₃⟩) (PFunctor.M.mk ⟨a, f₃⟩)
      ⊢ Eq (PFunctor.Approx.CofixA.intro a✝¹ fun j => (f✝¹ j).approx (HAdd.hAdd n 1) …
    -/
    have := mk_inj h₁
    /-
      case succ.step.step.f.f.refl.refl.refl
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      n_ih✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      a : F.A
      f₃ f₂ : F.B a → F.M
      hAgree₂ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₂ i)
      a✝¹ : F.A
      f✝¹ : F.B a✝¹ → F.M
      h₂ : Eq (PFunctor.M.mk ⟨a✝¹, f✝¹⟩) (PFunctor.M.mk ⟨a, f₂⟩)
      a✝ : F.A
      f✝ : F.B a✝ → F.M
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      f₁ : F.B a → F.M
      h₁ : Eq (PFunctor.M.mk ⟨a✝, f✝⟩) (PFunctor.M.mk ⟨a, f₁⟩)
      hAgree₁ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₁ i)
      clr : Eq (PFunctor.M.mk ⟨a, f₃⟩) (PFunctor.M.mk ⟨a, f₃⟩)
      this : Eq ⟨a✝, f✝⟩ ⟨a, f₁⟩
      ⊢ Eq (PFunctor.Approx.CofixA.intro a✝¹ fun j => (f✝¹ j).approx (HAdd.hAdd n 1) …
    -/
    cases this; clear h₁
    /-
      case succ.step.step.f.f.refl.refl.refl.refl
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      n_ih✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      a : F.A
      f₃ f₂ : F.B a → F.M
      hAgree₂ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₂ i)
      a✝ : F.A
      f✝ : F.B a✝ → F.M
      h₂ : Eq (PFunctor.M.mk ⟨a✝, f✝⟩) (PFunctor.M.mk ⟨a, f₂⟩)
      f₁ : F.B a → F.M
      hAgree₁ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₁ i)
      clr : Eq (PFunctor.M.mk ⟨a, f₃⟩) (PFunctor.M.mk ⟨a, f₃⟩)
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      ⊢ Eq (PFunctor.Approx.CofixA.intro a✝ fun j => (f✝ j).approx (HAdd.hAdd n 1))  …
    -/
    have := mk_inj h₂
    /-
      case succ.step.step.f.f.refl.refl.refl.refl
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      n_ih✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      a : F.A
      f₃ f₂ : F.B a → F.M
      hAgree₂ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₂ i)
      a✝ : F.A
      f✝ : F.B a✝ → F.M
      h₂ : Eq (PFunctor.M.mk ⟨a✝, f✝⟩) (PFunctor.M.mk ⟨a, f₂⟩)
      f₁ : F.B a → F.M
      hAgree₁ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₁ i)
      clr : Eq (PFunctor.M.mk ⟨a, f₃⟩) (PFunctor.M.mk ⟨a, f₃⟩)
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      this : Eq ⟨a✝, f✝⟩ ⟨a, f₂⟩
      ⊢ Eq (PFunctor.Approx.CofixA.intro a✝ fun j => (f✝ j).approx (HAdd.hAdd n 1))  …
    -/
    cases this; clear h₂

    /-
      case succ.step.step.f.f.refl.refl.refl.refl.refl
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      n_ih✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      a : F.A
      f₃ f₂ : F.B a → F.M
      hAgree₂ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₂ i)
      f₁ : F.B a → F.M
      hAgree₁ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₁ i)
      clr : Eq (PFunctor.M.mk ⟨a, f₃⟩) (PFunctor.M.mk ⟨a, f₃⟩)
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      ⊢ Eq (PFunctor.Approx.CofixA.intro a fun j => (f₂ j).approx (HAdd.hAdd n 1)) ( …
    -/
    congr
    /-
      case succ.step.step.f.f.refl.refl.refl.refl.refl.e_a
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      n_ih✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      a : F.A
      f₃ f₂ : F.B a → F.M
      hAgree₂ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₂ i)
      f₁ : F.B a → F.M
      hAgree₁ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₁ i)
      clr : Eq (PFunctor.M.mk ⟨a, f₃⟩) (PFunctor.M.mk ⟨a, f₃⟩)
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      ⊢ Eq (fun j => (f₂ j).approx (HAdd.hAdd n 1)) fun j => (f₁ j).approx (HAdd.hAd …
    -/
    ext i
    /-
      case succ.step.step.f.f.refl.refl.refl.refl.refl.e_a.h
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      n_ih✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      a : F.A
      f₃ f₂ : F.B a → F.M
      hAgree₂ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₂ i)
      f₁ : F.B a → F.M
      hAgree₁ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₁ i)
      clr : Eq (PFunctor.M.mk ⟨a, f₃⟩) (PFunctor.M.mk ⟨a, f₃⟩)
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      i : F.B a
      ⊢ Eq ((f₂ i).approx (HAdd.hAdd n 1)) ((f₁ i).approx (HAdd.hAdd n 1))
    -/
    apply n_ih
      /-
        case succ.step.step.f.f.refl.refl.refl.refl.refl.e_a.h.hx
        F : PFunctor.{u}
        inst✝ : Inhabited F.M
        n_ih✝ : DecidableEq F.A
        n : Nat
        n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
        a : F.A
        f₃ f₂ : F.B a → F.M
        hAgree₂ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₂ i)
        f₁ : F.B a → F.M
        hAgree₁ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₁ i)
        clr : Eq (PFunctor.M.mk ⟨a, f₃⟩) (PFunctor.M.mk ⟨a, f₃⟩)
        hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
        i : F.B a
        ⊢ PFunctor.M.Agree' n ?succ.step.step.f.f.refl.refl.refl.refl.refl.e_a.h.z (f₂ …
      -/
    · solve_by_elim
      /-
        🎉 no goals
      -/
      /-
        case succ.step.step.f.f.refl.refl.refl.refl.refl.e_a.h.hy
        F : PFunctor.{u}
        inst✝ : Inhabited F.M
        n_ih✝ : DecidableEq F.A
        n : Nat
        n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
        a : F.A
        f₃ f₂ : F.B a → F.M
        hAgree₂ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₂ i)
        f₁ : F.B a → F.M
        hAgree₁ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₁ i)
        clr : Eq (PFunctor.M.mk ⟨a, f₃⟩) (PFunctor.M.mk ⟨a, f₃⟩)
        hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
        i : F.B a
        ⊢ PFunctor.M.Agree' n (f₃ i) (f₁ i)
      -/
    · solve_by_elim
      /-
        🎉 no goals
      -/
    /-
      case succ.step.step.f.f.refl.refl.refl.refl.refl.e_a.h.hrec
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      n_ih✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      a : F.A
      f₃ f₂ : F.B a → F.M
      hAgree₂ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₂ i)
      f₁ : F.B a → F.M
      hAgree₁ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₁ i)
      clr : Eq (PFunctor.M.mk ⟨a, f₃⟩) (PFunctor.M.mk ⟨a, f₃⟩)
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      i : F.B a
      ⊢ ∀ (ps : PFunctor.Approx.Path F), Eq n (List.length ps) → Eq (PFunctor.M.isel …
    -/
    introv h
    /-
      case succ.step.step.f.f.refl.refl.refl.refl.refl.e_a.h.hrec
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      n_ih✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      a : F.A
      f₃ f₂ : F.B a → F.M
      hAgree₂ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₂ i)
      f₁ : F.B a → F.M
      hAgree₁ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₁ i)
      clr : Eq (PFunctor.M.mk ⟨a, f₃⟩) (PFunctor.M.mk ⟨a, f₃⟩)
      hrec : ∀ (ps : PFunctor.Approx.Path F), Eq (HAdd.hAdd n 1) (List.length ps) →  …
      i : F.B a
      ps : PFunctor.Approx.Path F
      h : Eq n (List.length ps)
      ⊢ Eq (PFunctor.M.iselect ps (f₂ i)) (PFunctor.M.iselect ps (f₁ i))
    -/
    specialize hrec (⟨_, i⟩ :: ps) (congr_arg _ h)
    /-
      case succ.step.step.f.f.refl.refl.refl.refl.refl.e_a.h.hrec
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      n_ih✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      a : F.A
      f₃ f₂ : F.B a → F.M
      hAgree₂ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₂ i)
      f₁ : F.B a → F.M
      hAgree₁ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₁ i)
      clr : Eq (PFunctor.M.mk ⟨a, f₃⟩) (PFunctor.M.mk ⟨a, f₃⟩)
      i : F.B a
      ps : PFunctor.Approx.Path F
      h : Eq n (List.length ps)
      hrec : Eq (PFunctor.M.iselect (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f₂⟩)) ( …
      ⊢ Eq (PFunctor.M.iselect ps (f₂ i)) (PFunctor.M.iselect ps (f₁ i))
    -/
    simp only [iselect_cons] at hrec
    /-
      case succ.step.step.f.f.refl.refl.refl.refl.refl.e_a.h.hrec
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      n_ih✝ : DecidableEq F.A
      n : Nat
      n_ih : ∀ (x y z : F.M), PFunctor.M.Agree' n z x → PFunctor.M.Agree' n z y → (∀ …
      a : F.A
      f₃ f₂ : F.B a → F.M
      hAgree₂ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₂ i)
      f₁ : F.B a → F.M
      hAgree₁ : ∀ (i : F.B a), PFunctor.M.Agree' n (f₃ i) (f₁ i)
      clr : Eq (PFunctor.M.mk ⟨a, f₃⟩) (PFunctor.M.mk ⟨a, f₃⟩)
      i : F.B a
      ps : PFunctor.Approx.Path F
      h : Eq n (List.length ps)
      hrec : Eq (PFunctor.M.iselect ps (f₂ i)) (PFunctor.M.iselect ps (f₁ i))
      ⊢ Eq (PFunctor.M.iselect ps (f₂ i)) (PFunctor.M.iselect ps (f₁ i))
    -/
    exact hrec
    /-
      🎉 no goals
    -/


theorem ext [Inhabited (M F)] (x y : M F) (H : ∀ ps : Path F, iselect ps x = iselect ps y) :
    x = y := by
  /-
    F : PFunctor.{u}
    inst✝ : Inhabited F.M
    x y : F.M
    H : ∀ (ps : PFunctor.Approx.Path F), Eq (PFunctor.M.iselect ps x) (PFunctor.M. …
    ⊢ Eq x y
  -/
  apply ext'; intro i
  /-
    case H
    F : PFunctor.{u}
    inst✝ : Inhabited F.M
    x y : F.M
    H : ∀ (ps : PFunctor.Approx.Path F), Eq (PFunctor.M.iselect ps x) (PFunctor.M. …
    i : Nat
    ⊢ Eq (x.approx i) (y.approx i)
  -/
  induction' i with i i_ih
    /-
      case H.zero
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      x y : F.M
      H : ∀ (ps : PFunctor.Approx.Path F), Eq (PFunctor.M.iselect ps x) (PFunctor.M. …
      ⊢ Eq (x.approx 0) (y.approx 0)
    -/
  · cases x.approx 0
    /-
      case H.zero.continue
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      x y : F.M
      H : ∀ (ps : PFunctor.Approx.Path F), Eq (PFunctor.M.iselect ps x) (PFunctor.M. …
      ⊢ Eq PFunctor.Approx.CofixA.continue (y.approx 0)
    -/
    cases y.approx 0
    /-
      case H.zero.continue.continue
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      x y : F.M
      H : ∀ (ps : PFunctor.Approx.Path F), Eq (PFunctor.M.iselect ps x) (PFunctor.M. …
      ⊢ Eq PFunctor.Approx.CofixA.continue PFunctor.Approx.CofixA.continue
    -/
    constructor
    /-
      🎉 no goals
    -/
    /-
      case H.succ
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      x y : F.M
      H : ∀ (ps : PFunctor.Approx.Path F), Eq (PFunctor.M.iselect ps x) (PFunctor.M. …
      i : Nat
      i_ih : Eq (x.approx i) (y.approx i)
      ⊢ Eq (x.approx (HAdd.hAdd i 1)) (y.approx (HAdd.hAdd i 1))
    -/
  · apply ext_aux x y x
      /-
        case H.succ.hx
        F : PFunctor.{u}
        inst✝ : Inhabited F.M
        x y : F.M
        H : ∀ (ps : PFunctor.Approx.Path F), Eq (PFunctor.M.iselect ps x) (PFunctor.M. …
        i : Nat
        i_ih : Eq (x.approx i) (y.approx i)
        ⊢ PFunctor.M.Agree' i x x
      -/
    · rw [← agree_iff_agree']
      /-
        case H.succ.hx
        F : PFunctor.{u}
        inst✝ : Inhabited F.M
        x y : F.M
        H : ∀ (ps : PFunctor.Approx.Path F), Eq (PFunctor.M.iselect ps x) (PFunctor.M. …
        i : Nat
        i_ih : Eq (x.approx i) (y.approx i)
        ⊢ PFunctor.Approx.Agree (x.approx i) (x.approx (HAdd.hAdd i 1))
      -/
      apply x.consistent
      /-
        🎉 no goals
      -/
      /-
        case H.succ.hy
        F : PFunctor.{u}
        inst✝ : Inhabited F.M
        x y : F.M
        H : ∀ (ps : PFunctor.Approx.Path F), Eq (PFunctor.M.iselect ps x) (PFunctor.M. …
        i : Nat
        i_ih : Eq (x.approx i) (y.approx i)
        ⊢ PFunctor.M.Agree' i x y
      -/
    · rw [← agree_iff_agree', i_ih]
      /-
        case H.succ.hy
        F : PFunctor.{u}
        inst✝ : Inhabited F.M
        x y : F.M
        H : ∀ (ps : PFunctor.Approx.Path F), Eq (PFunctor.M.iselect ps x) (PFunctor.M. …
        i : Nat
        i_ih : Eq (x.approx i) (y.approx i)
        ⊢ PFunctor.Approx.Agree (y.approx i) (y.approx (HAdd.hAdd i 1))
      -/
      apply y.consistent
      /-
        🎉 no goals
      -/
    /-
      case H.succ.hrec
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      x y : F.M
      H : ∀ (ps : PFunctor.Approx.Path F), Eq (PFunctor.M.iselect ps x) (PFunctor.M. …
      i : Nat
      i_ih : Eq (x.approx i) (y.approx i)
      ⊢ ∀ (ps : PFunctor.Approx.Path F), Eq i (List.length ps) → Eq (PFunctor.M.isel …
    -/
    introv H'
    /-
      case H.succ.hrec
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      x y : F.M
      H : ∀ (ps : PFunctor.Approx.Path F), Eq (PFunctor.M.iselect ps x) (PFunctor.M. …
      i : Nat
      i_ih : Eq (x.approx i) (y.approx i)
      ps : PFunctor.Approx.Path F
      H' : Eq i (List.length ps)
      ⊢ Eq (PFunctor.M.iselect ps x) (PFunctor.M.iselect ps y)
    -/
    dsimp only [iselect] at H
    /-
      case H.succ.hrec
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      x y : F.M
      H : ∀ (ps : PFunctor.Approx.Path F), Eq (PFunctor.M.isubtree ps x).head (PFunc …
      i : Nat
      i_ih : Eq (x.approx i) (y.approx i)
      ps : PFunctor.Approx.Path F
      H' : Eq i (List.length ps)
      ⊢ Eq (PFunctor.M.iselect ps x) (PFunctor.M.iselect ps y)
    -/
    cases H'
    /-
      case H.succ.hrec.refl
      F : PFunctor.{u}
      inst✝ : Inhabited F.M
      x y : F.M
      H : ∀ (ps : PFunctor.Approx.Path F), Eq (PFunctor.M.isubtree ps x).head (PFunc …
      ps : PFunctor.Approx.Path F
      i_ih : Eq (x.approx (List.length ps)) (y.approx (List.length ps))
      ⊢ Eq (PFunctor.M.iselect ps x) (PFunctor.M.iselect ps y)
    -/
    apply H ps
    /-
      🎉 no goals
    -/


local infixl:50 " ~ " => R


/-- Bisimulation is the standard proof technique for equality between
infinite tree-like structures -/
structure IsBisimulation : Prop where
  /-- The head of the trees are equal -/
  head : ∀ {a a'} {f f'}, M.mk ⟨a, f⟩ ~ M.mk ⟨a', f'⟩ → a = a'
  /-- The tails are equal -/
  tail : ∀ {a} {f f' : F.B a → M F}, M.mk ⟨a, f⟩ ~ M.mk ⟨a, f'⟩ → ∀ i : F.B a, f i ~ f' i


theorem nth_of_bisim [Inhabited (M F)] (bisim : IsBisimulation R) (s₁ s₂) (ps : Path F) :
    (R s₁ s₂) →
      IsPath ps s₁ ∨ IsPath ps s₂ →
        iselect ps s₁ = iselect ps s₂ ∧
          ∃ (a : _) (f f' : F.B a → M F),
            isubtree ps s₁ = M.mk ⟨a, f⟩ ∧
              isubtree ps s₂ = M.mk ⟨a, f'⟩ ∧ ∀ i : F.B a, f i ~ f' i := by
  /-
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    s₁ s₂ : F.M
    ps : PFunctor.Approx.Path F
    ⊢ R s₁ s₂ → Or (PFunctor.M.IsPath ps s₁) (PFunctor.M.IsPath ps s₂) → And (Eq ( …
  -/
  intro h₀ hh
  /-
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    s₁ s₂ : F.M
    ps : PFunctor.Approx.Path F
    h₀ : R s₁ s₂
    hh : Or (PFunctor.M.IsPath ps s₁) (PFunctor.M.IsPath ps s₂)
    ⊢ And (Eq (PFunctor.M.iselect ps s₁) (PFunctor.M.iselect ps s₂)) (Exists fun a …
  -/
  induction' s₁ using PFunctor.M.casesOn' with a f
  /-
    case f
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    s₂ : F.M
    ps : PFunctor.Approx.Path F
    a : F.A
    f : F.B a → F.M
    h₀ : R (PFunctor.M.mk ⟨a, f⟩) s₂
    hh : Or (PFunctor.M.IsPath ps (PFunctor.M.mk ⟨a, f⟩)) (PFunctor.M.IsPath ps s₂)
    ⊢ And (Eq (PFunctor.M.iselect ps (PFunctor.M.mk ⟨a, f⟩)) (PFunctor.M.iselect p …
  -/
  induction' s₂ using PFunctor.M.casesOn' with a' f'
  /-
    case f.f
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    ps : PFunctor.Approx.Path F
    a : F.A
    f : F.B a → F.M
    a' : F.A
    f' : F.B a' → F.M
    h₀ : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a', f'⟩)
    hh : Or (PFunctor.M.IsPath ps (PFunctor.M.mk ⟨a, f⟩)) (PFunctor.M.IsPath ps (P …
    ⊢ And (Eq (PFunctor.M.iselect ps (PFunctor.M.mk ⟨a, f⟩)) (PFunctor.M.iselect p …
  -/
  obtain rfl : a = a' := bisim.head h₀
  /-
    case f.f
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    ps : PFunctor.Approx.Path F
    a : F.A
    f f' : F.B a → F.M
    h₀ : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
    hh : Or (PFunctor.M.IsPath ps (PFunctor.M.mk ⟨a, f⟩)) (PFunctor.M.IsPath ps (P …
    ⊢ And (Eq (PFunctor.M.iselect ps (PFunctor.M.mk ⟨a, f⟩)) (PFunctor.M.iselect p …
  -/
  induction' ps with i ps ps_ih generalizing a f f'
    /-
      case f.f.nil
      F : PFunctor.{u}
      R : F.M → F.M → Prop
      inst✝ : Inhabited F.M
      bisim : PFunctor.M.IsBisimulation R
      a : F.A
      f f' : F.B a → F.M
      h₀ : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
      hh : Or (PFunctor.M.IsPath List.nil (PFunctor.M.mk ⟨a, f⟩)) (PFunctor.M.IsPath …
      ⊢ And (Eq (PFunctor.M.iselect List.nil (PFunctor.M.mk ⟨a, f⟩)) (PFunctor.M.ise …
    -/
  · exists rfl, a, f, f', rfl, rfl
    /-
      case f.f.nil
      F : PFunctor.{u}
      R : F.M → F.M → Prop
      inst✝ : Inhabited F.M
      bisim : PFunctor.M.IsBisimulation R
      a : F.A
      f f' : F.B a → F.M
      h₀ : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
      hh : Or (PFunctor.M.IsPath List.nil (PFunctor.M.mk ⟨a, f⟩)) (PFunctor.M.IsPath …
      ⊢ ∀ (i : F.B a), R (f i) (f' i)
    -/
    apply bisim.tail h₀
    /-
      🎉 no goals
    -/
  /-
    case f.f.cons
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    i : F.Idx
    ps : List F.Idx
    ps_ih : ∀ (a : F.A) (f f' : F.B a → F.M), R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M …
    a : F.A
    f f' : F.B a → F.M
    h₀ : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
    hh : Or (PFunctor.M.IsPath (List.cons i ps) (PFunctor.M.mk ⟨a, f⟩)) (PFunctor. …
    ⊢ And (Eq (PFunctor.M.iselect (List.cons i ps) (PFunctor.M.mk ⟨a, f⟩)) (PFunct …
  -/
  cases' i with a' i
  /-
    case f.f.cons.mk
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    ps : List F.Idx
    ps_ih : ∀ (a : F.A) (f f' : F.B a → F.M), R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M …
    a : F.A
    f f' : F.B a → F.M
    h₀ : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
    a' : F.A
    i : F.B a'
    hh : Or (PFunctor.M.IsPath (List.cons ⟨a', i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)) (PFu …
    ⊢ And (Eq (PFunctor.M.iselect (List.cons ⟨a', i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)) ( …
  -/
  obtain rfl : a = a' := by rcases hh with hh|hh <;> cases isPath_cons hh <;> rfl
  /-
    case f.f.cons.mk
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    ps : List F.Idx
    ps_ih : ∀ (a : F.A) (f f' : F.B a → F.M), R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M …
    a : F.A
    f f' : F.B a → F.M
    h₀ : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
    i : F.B a
    hh : Or (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)) (PFun …
    ⊢ And (Eq (PFunctor.M.iselect (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)) (P …
  -/
  dsimp only [iselect] at ps_ih ⊢
  /-
    case f.f.cons.mk
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    ps : List F.Idx
    ps_ih : ∀ (a : F.A) (f f' : F.B a → F.M), R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M …
    a : F.A
    f f' : F.B a → F.M
    h₀ : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
    i : F.B a
    hh : Or (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)) (PFun …
    ⊢ And (Eq (PFunctor.M.isubtree (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)).h …
  -/
  have h₁ := bisim.tail h₀ i
  /-
    case f.f.cons.mk
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    ps : List F.Idx
    ps_ih : ∀ (a : F.A) (f f' : F.B a → F.M), R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M …
    a : F.A
    f f' : F.B a → F.M
    h₀ : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
    i : F.B a
    hh : Or (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)) (PFun …
    h₁ : R (f i) (f' i)
    ⊢ And (Eq (PFunctor.M.isubtree (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)).h …
  -/
  induction' h : f i using PFunctor.M.casesOn' with a₀ f₀
  /-
    case f.f.cons.mk.f
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    ps : List F.Idx
    ps_ih : ∀ (a : F.A) (f f' : F.B a → F.M), R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M …
    a : F.A
    f f' : F.B a → F.M
    h₀ : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
    i : F.B a
    hh : Or (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)) (PFun …
    h₁ : R (f i) (f' i)
    a₀ : F.A
    f₀ : F.B a₀ → F.M
    h : Eq (f i) (PFunctor.M.mk ⟨a₀, f₀⟩)
    ⊢ And (Eq (PFunctor.M.isubtree (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)).h …
  -/
  induction' h' : f' i using PFunctor.M.casesOn' with a₁ f₁
  /-
    case f.f.cons.mk.f.f
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    ps : List F.Idx
    ps_ih : ∀ (a : F.A) (f f' : F.B a → F.M), R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M …
    a : F.A
    f f' : F.B a → F.M
    h₀ : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
    i : F.B a
    hh : Or (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)) (PFun …
    h₁ : R (f i) (f' i)
    a₀ : F.A
    f₀ : F.B a₀ → F.M
    h : Eq (f i) (PFunctor.M.mk ⟨a₀, f₀⟩)
    a₁ : F.A
    f₁ : F.B a₁ → F.M
    h' : Eq (f' i) (PFunctor.M.mk ⟨a₁, f₁⟩)
    ⊢ And (Eq (PFunctor.M.isubtree (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)).h …
  -/
  simp only [h, h', isubtree_cons] at ps_ih ⊢
  /-
    case f.f.cons.mk.f.f
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    ps : List F.Idx
    ps_ih : ∀ (a : F.A) (f f' : F.B a → F.M), R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M …
    a : F.A
    f f' : F.B a → F.M
    h₀ : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
    i : F.B a
    hh : Or (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)) (PFun …
    h₁ : R (f i) (f' i)
    a₀ : F.A
    f₀ : F.B a₀ → F.M
    h : Eq (f i) (PFunctor.M.mk ⟨a₀, f₀⟩)
    a₁ : F.A
    f₁ : F.B a₁ → F.M
    h' : Eq (f' i) (PFunctor.M.mk ⟨a₁, f₁⟩)
    ⊢ And (Eq (PFunctor.M.isubtree ps (PFunctor.M.mk ⟨a₀, f₀⟩)).head (PFunctor.M.i …
  -/
  rw [h, h'] at h₁
  /-
    case f.f.cons.mk.f.f
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    ps : List F.Idx
    ps_ih : ∀ (a : F.A) (f f' : F.B a → F.M), R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M …
    a : F.A
    f f' : F.B a → F.M
    h₀ : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
    i : F.B a
    hh : Or (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)) (PFun …
    a₀ : F.A
    f₀ : F.B a₀ → F.M
    h : Eq (f i) (PFunctor.M.mk ⟨a₀, f₀⟩)
    a₁ : F.A
    f₁ : F.B a₁ → F.M
    h₁ : R (PFunctor.M.mk ⟨a₀, f₀⟩) (PFunctor.M.mk ⟨a₁, f₁⟩)
    h' : Eq (f' i) (PFunctor.M.mk ⟨a₁, f₁⟩)
    ⊢ And (Eq (PFunctor.M.isubtree ps (PFunctor.M.mk ⟨a₀, f₀⟩)).head (PFunctor.M.i …
  -/
  obtain rfl : a₀ = a₁ := bisim.head h₁
  /-
    case f.f.cons.mk.f.f
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    ps : List F.Idx
    ps_ih : ∀ (a : F.A) (f f' : F.B a → F.M), R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M …
    a : F.A
    f f' : F.B a → F.M
    h₀ : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
    i : F.B a
    hh : Or (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)) (PFun …
    a₀ : F.A
    f₀ : F.B a₀ → F.M
    h : Eq (f i) (PFunctor.M.mk ⟨a₀, f₀⟩)
    f₁ : F.B a₀ → F.M
    h₁ : R (PFunctor.M.mk ⟨a₀, f₀⟩) (PFunctor.M.mk ⟨a₀, f₁⟩)
    h' : Eq (f' i) (PFunctor.M.mk ⟨a₀, f₁⟩)
    ⊢ And (Eq (PFunctor.M.isubtree ps (PFunctor.M.mk ⟨a₀, f₀⟩)).head (PFunctor.M.i …
  -/
  apply ps_ih _ _ _ h₁
  /-
    case f.f.cons.mk.f.f
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    ps : List F.Idx
    ps_ih : ∀ (a : F.A) (f f' : F.B a → F.M), R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M …
    a : F.A
    f f' : F.B a → F.M
    h₀ : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
    i : F.B a
    hh : Or (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)) (PFun …
    a₀ : F.A
    f₀ : F.B a₀ → F.M
    h : Eq (f i) (PFunctor.M.mk ⟨a₀, f₀⟩)
    f₁ : F.B a₀ → F.M
    h₁ : R (PFunctor.M.mk ⟨a₀, f₀⟩) (PFunctor.M.mk ⟨a₀, f₁⟩)
    h' : Eq (f' i) (PFunctor.M.mk ⟨a₀, f₁⟩)
    ⊢ Or (PFunctor.M.IsPath ps (PFunctor.M.mk ⟨a₀, f₀⟩)) (PFunctor.M.IsPath ps (PF …
  -/
  rw [← h, ← h']
  /-
    case f.f.cons.mk.f.f
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Inhabited F.M
    bisim : PFunctor.M.IsBisimulation R
    ps : List F.Idx
    ps_ih : ∀ (a : F.A) (f f' : F.B a → F.M), R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M …
    a : F.A
    f f' : F.B a → F.M
    h₀ : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
    i : F.B a
    hh : Or (PFunctor.M.IsPath (List.cons ⟨a, i⟩ ps) (PFunctor.M.mk ⟨a, f⟩)) (PFun …
    a₀ : F.A
    f₀ : F.B a₀ → F.M
    h : Eq (f i) (PFunctor.M.mk ⟨a₀, f₀⟩)
    f₁ : F.B a₀ → F.M
    h₁ : R (PFunctor.M.mk ⟨a₀, f₀⟩) (PFunctor.M.mk ⟨a₀, f₁⟩)
    h' : Eq (f' i) (PFunctor.M.mk ⟨a₀, f₁⟩)
    ⊢ Or (PFunctor.M.IsPath ps (f i)) (PFunctor.M.IsPath ps (f' i))
  -/
  apply Or.imp isPath_cons' isPath_cons' hh
  /-
    🎉 no goals
  -/


theorem eq_of_bisim [Nonempty (M F)] (bisim : IsBisimulation R) : ∀ s₁ s₂, R s₁ s₂ → s₁ = s₂ := by
  /-
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Nonempty F.M
    bisim : PFunctor.M.IsBisimulation R
    ⊢ ∀ (s₁ s₂ : F.M), R s₁ s₂ → Eq s₁ s₂
  -/
  inhabit M F
  /-
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Nonempty F.M
    bisim : PFunctor.M.IsBisimulation R
    inhabited_h : Inhabited F.M
    ⊢ ∀ (s₁ s₂ : F.M), R s₁ s₂ → Eq s₁ s₂
  -/
  introv Hr; apply ext
  /-
    case H
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Nonempty F.M
    bisim : PFunctor.M.IsBisimulation R
    inhabited_h : Inhabited F.M
    s₁ s₂ : F.M
    Hr : R s₁ s₂
    ⊢ ∀ (ps : PFunctor.Approx.Path F), Eq (PFunctor.M.iselect ps s₁) (PFunctor.M.i …
  -/
  introv
  /-
    case H
    F : PFunctor.{u}
    R : F.M → F.M → Prop
    inst✝ : Nonempty F.M
    bisim : PFunctor.M.IsBisimulation R
    inhabited_h : Inhabited F.M
    s₁ s₂ : F.M
    Hr : R s₁ s₂
    ps : PFunctor.Approx.Path F
    ⊢ Eq (PFunctor.M.iselect ps s₁) (PFunctor.M.iselect ps s₂)
  -/
  by_cases h : IsPath ps s₁ ∨ IsPath ps s₂
    /-
      case pos
      F : PFunctor.{u}
      R : F.M → F.M → Prop
      inst✝ : Nonempty F.M
      bisim : PFunctor.M.IsBisimulation R
      inhabited_h : Inhabited F.M
      s₁ s₂ : F.M
      Hr : R s₁ s₂
      ps : PFunctor.Approx.Path F
      h : Or (PFunctor.M.IsPath ps s₁) (PFunctor.M.IsPath ps s₂)
      ⊢ Eq (PFunctor.M.iselect ps s₁) (PFunctor.M.iselect ps s₂)
    -/
  · have H := nth_of_bisim R bisim _ _ ps Hr h
    /-
      case pos
      F : PFunctor.{u}
      R : F.M → F.M → Prop
      inst✝ : Nonempty F.M
      bisim : PFunctor.M.IsBisimulation R
      inhabited_h : Inhabited F.M
      s₁ s₂ : F.M
      Hr : R s₁ s₂
      ps : PFunctor.Approx.Path F
      h : Or (PFunctor.M.IsPath ps s₁) (PFunctor.M.IsPath ps s₂)
      H : And (Eq (PFunctor.M.iselect ps s₁) (PFunctor.M.iselect ps s₂)) (Exists fun …
      ⊢ Eq (PFunctor.M.iselect ps s₁) (PFunctor.M.iselect ps s₂)
    -/
    exact H.left
    /-
      🎉 no goals
    -/
    /-
      case neg
      F : PFunctor.{u}
      R : F.M → F.M → Prop
      inst✝ : Nonempty F.M
      bisim : PFunctor.M.IsBisimulation R
      inhabited_h : Inhabited F.M
      s₁ s₂ : F.M
      Hr : R s₁ s₂
      ps : PFunctor.Approx.Path F
      h : Not (Or (PFunctor.M.IsPath ps s₁) (PFunctor.M.IsPath ps s₂))
      ⊢ Eq (PFunctor.M.iselect ps s₁) (PFunctor.M.iselect ps s₂)
    -/
  · rw [not_or] at h
    /-
      case neg
      F : PFunctor.{u}
      R : F.M → F.M → Prop
      inst✝ : Nonempty F.M
      bisim : PFunctor.M.IsBisimulation R
      inhabited_h : Inhabited F.M
      s₁ s₂ : F.M
      Hr : R s₁ s₂
      ps : PFunctor.Approx.Path F
      h : And (Not (PFunctor.M.IsPath ps s₁)) (Not (PFunctor.M.IsPath ps s₂))
      ⊢ Eq (PFunctor.M.iselect ps s₁) (PFunctor.M.iselect ps s₂)
    -/
    cases' h with h₀ h₁
    /-
      case neg.intro
      F : PFunctor.{u}
      R : F.M → F.M → Prop
      inst✝ : Nonempty F.M
      bisim : PFunctor.M.IsBisimulation R
      inhabited_h : Inhabited F.M
      s₁ s₂ : F.M
      Hr : R s₁ s₂
      ps : PFunctor.Approx.Path F
      h₀ : Not (PFunctor.M.IsPath ps s₁)
      h₁ : Not (PFunctor.M.IsPath ps s₂)
      ⊢ Eq (PFunctor.M.iselect ps s₁) (PFunctor.M.iselect ps s₂)
    -/
    simp only [iselect_eq_default, *, not_false_iff]
    /-
      🎉 no goals
    -/


/-- corecursor for `M F` with swapped arguments -/
def corecOn {X : Type*} (x₀ : X) (f : X → F X) : M F :=
  M.corec f x₀


theorem dest_corec (g : α → P α) (x : α) : M.dest (M.corec g x) = P.map (M.corec g) (g x) := by
  /-
    P : PFunctor.{u}
    α : Type u_2
    g : α → ↑P α
    x : α
    ⊢ Eq (PFunctor.M.corec g x).dest (P.map (PFunctor.M.corec g) (g x))
  -/
  rw [corec_def, dest_mk]
  /-
    🎉 no goals
  -/


theorem bisim (R : M P → M P → Prop)
    (h : ∀ x y, R x y → ∃ a f f', M.dest x = ⟨a, f⟩ ∧ M.dest y = ⟨a, f'⟩ ∧ ∀ i, R (f i) (f' i)) :
    ∀ x y, R x y → x = y := by
  /-
    P : PFunctor.{u}
    R : P.M → P.M → Prop
    h : ∀ (x y : P.M), R x y → Exists fun a => Exists fun f => Exists fun f' => An …
    ⊢ ∀ (x y : P.M), R x y → Eq x y
  -/
  introv h'
  /-
    P : PFunctor.{u}
    R : P.M → P.M → Prop
    h : ∀ (x y : P.M), R x y → Exists fun a => Exists fun f => Exists fun f' => An …
    x y : P.M
    h' : R x y
    ⊢ Eq x y
  -/
  haveI := Inhabited.mk x.head
  /-
    P : PFunctor.{u}
    R : P.M → P.M → Prop
    h : ∀ (x y : P.M), R x y → Exists fun a => Exists fun f => Exists fun f' => An …
    x y : P.M
    h' : R x y
    this : Inhabited P.A
    ⊢ Eq x y
  -/
  apply eq_of_bisim R _ _ _ h'; clear h' x y
  /-
    P : PFunctor.{u}
    R : P.M → P.M → Prop
    h : ∀ (x y : P.M), R x y → Exists fun a => Exists fun f => Exists fun f' => An …
    this : Inhabited P.A
    ⊢ PFunctor.M.IsBisimulation R
  -/
  constructor <;> introv ih <;> rcases h _ _ ih with ⟨a'', g, g', h₀, h₁, h₂⟩ <;> clear h
    /-
      case head.intro.intro.intro.intro.intro
      P : PFunctor.{u}
      R : P.M → P.M → Prop
      this : Inhabited P.A
      a a' : P.A
      f : P.B a → P.M
      f' : P.B a' → P.M
      ih : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a', f'⟩)
      a'' : P.A
      g g' : P.B a'' → P.M
      h₀ : Eq (PFunctor.M.mk ⟨a, f⟩).dest ⟨a'', g⟩
      h₁ : Eq (PFunctor.M.mk ⟨a', f'⟩).dest ⟨a'', g'⟩
      h₂ : ∀ (i : P.B a''), R (g i) (g' i)
      ⊢ Eq a a'
    -/
  · replace h₀ := congr_arg Sigma.fst h₀
    /-
      case head.intro.intro.intro.intro.intro
      P : PFunctor.{u}
      R : P.M → P.M → Prop
      this : Inhabited P.A
      a a' : P.A
      f : P.B a → P.M
      f' : P.B a' → P.M
      ih : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a', f'⟩)
      a'' : P.A
      g g' : P.B a'' → P.M
      h₁ : Eq (PFunctor.M.mk ⟨a', f'⟩).dest ⟨a'', g'⟩
      h₂ : ∀ (i : P.B a''), R (g i) (g' i)
      h₀ : Eq (PFunctor.M.mk ⟨a, f⟩).dest.fst ⟨a'', g⟩.fst
      ⊢ Eq a a'
    -/
    replace h₁ := congr_arg Sigma.fst h₁
    /-
      case head.intro.intro.intro.intro.intro
      P : PFunctor.{u}
      R : P.M → P.M → Prop
      this : Inhabited P.A
      a a' : P.A
      f : P.B a → P.M
      f' : P.B a' → P.M
      ih : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a', f'⟩)
      a'' : P.A
      g g' : P.B a'' → P.M
      h₂ : ∀ (i : P.B a''), R (g i) (g' i)
      h₀ : Eq (PFunctor.M.mk ⟨a, f⟩).dest.fst ⟨a'', g⟩.fst
      h₁ : Eq (PFunctor.M.mk ⟨a', f'⟩).dest.fst ⟨a'', g'⟩.fst
      ⊢ Eq a a'
    -/
    simp only [dest_mk] at h₀ h₁
    /-
      case head.intro.intro.intro.intro.intro
      P : PFunctor.{u}
      R : P.M → P.M → Prop
      this : Inhabited P.A
      a a' : P.A
      f : P.B a → P.M
      f' : P.B a' → P.M
      ih : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a', f'⟩)
      a'' : P.A
      g g' : P.B a'' → P.M
      h₂ : ∀ (i : P.B a''), R (g i) (g' i)
      h₀ : Eq a a''
      h₁ : Eq a' a''
      ⊢ Eq a a'
    -/
    rw [h₀, h₁]
    /-
      🎉 no goals
    -/
    /-
      case tail.intro.intro.intro.intro.intro
      P : PFunctor.{u}
      R : P.M → P.M → Prop
      this : Inhabited P.A
      a : P.A
      f f' : P.B a → P.M
      ih : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
      i : P.B a
      a'' : P.A
      g g' : P.B a'' → P.M
      h₀ : Eq (PFunctor.M.mk ⟨a, f⟩).dest ⟨a'', g⟩
      h₁ : Eq (PFunctor.M.mk ⟨a, f'⟩).dest ⟨a'', g'⟩
      h₂ : ∀ (i : P.B a''), R (g i) (g' i)
      ⊢ R (f i) (f' i)
    -/
  · simp only [dest_mk] at h₀ h₁
    /-
      case tail.intro.intro.intro.intro.intro
      P : PFunctor.{u}
      R : P.M → P.M → Prop
      this : Inhabited P.A
      a : P.A
      f f' : P.B a → P.M
      ih : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
      i : P.B a
      a'' : P.A
      g g' : P.B a'' → P.M
      h₀ : Eq ⟨a, f⟩ ⟨a'', g⟩
      h₁ : Eq ⟨a, f'⟩ ⟨a'', g'⟩
      h₂ : ∀ (i : P.B a''), R (g i) (g' i)
      ⊢ R (f i) (f' i)
    -/
    cases h₀
    /-
      case tail.intro.intro.intro.intro.intro.refl
      P : PFunctor.{u}
      R : P.M → P.M → Prop
      this : Inhabited P.A
      a : P.A
      f f' : P.B a → P.M
      ih : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
      i : P.B a
      g' : P.B a → P.M
      h₁ : Eq ⟨a, f'⟩ ⟨a, g'⟩
      h₂ : ∀ (i : P.B a), R (f i) (g' i)
      ⊢ R (f i) (f' i)
    -/
    cases h₁
    /-
      case tail.intro.intro.intro.intro.intro.refl.refl
      P : PFunctor.{u}
      R : P.M → P.M → Prop
      this : Inhabited P.A
      a : P.A
      f f' : P.B a → P.M
      ih : R (PFunctor.M.mk ⟨a, f⟩) (PFunctor.M.mk ⟨a, f'⟩)
      i : P.B a
      h₂ : ∀ (i : P.B a), R (f i) (f' i)
      ⊢ R (f i) (f' i)
    -/
    apply h₂
    /-
      🎉 no goals
    -/


theorem bisim' {α : Type*} (Q : α → Prop) (u v : α → M P)
    (h : ∀ x, Q x → ∃ a f f',
          M.dest (u x) = ⟨a, f⟩
          ∧ M.dest (v x) = ⟨a, f'⟩
          ∧ ∀ i, ∃ x', Q x' ∧ f i = u x' ∧ f' i = v x'
      ) :
    ∀ x, Q x → u x = v x := fun x Qx =>
  let R := fun w z : M P => ∃ x', Q x' ∧ w = u x' ∧ z = v x'
  @M.bisim P R
    (fun _ _ ⟨x', Qx', xeq, yeq⟩ =>
      let ⟨a, f, f', ux'eq, vx'eq, h'⟩ := h x' Qx'
      ⟨a, f, f', xeq.symm ▸ ux'eq, yeq.symm ▸ vx'eq, h'⟩)
    _ _ ⟨x, Qx, rfl, rfl⟩

-- for the record, show M_bisim follows from _bisim'

theorem bisim_equiv (R : M P → M P → Prop)
    (h : ∀ x y, R x y → ∃ a f f', M.dest x = ⟨a, f⟩ ∧ M.dest y = ⟨a, f'⟩ ∧ ∀ i, R (f i) (f' i)) :
    ∀ x y, R x y → x = y := fun x y Rxy =>
  let Q : M P × M P → Prop := fun p => R p.fst p.snd
  bisim' Q Prod.fst Prod.snd
    (fun p Qp =>
      let ⟨a, f, f', hx, hy, h'⟩ := h p.fst p.snd Qp
      ⟨a, f, f', hx, hy, fun i => ⟨⟨f i, f' i⟩, h' i, rfl, rfl⟩⟩)
    ⟨x, y⟩ Rxy


theorem corec_unique (g : α → P α) (f : α → M P) (hyp : ∀ x, M.dest (f x) = P.map f (g x)) :
    f = M.corec g := by
  /-
    P : PFunctor.{u}
    α : Type u_2
    g : α → ↑P α
    f : α → P.M
    hyp : ∀ (x : α), Eq (f x).dest (P.map f (g x))
    ⊢ Eq f (PFunctor.M.corec g)
  -/
  ext x
  /-
    case h
    P : PFunctor.{u}
    α : Type u_2
    g : α → ↑P α
    f : α → P.M
    hyp : ∀ (x : α), Eq (f x).dest (P.map f (g x))
    x : α
    ⊢ Eq (f x) (PFunctor.M.corec g x)
  -/
  apply bisim' (fun _ => True) _ _ _ _ trivial
  /-
    P : PFunctor.{u}
    α : Type u_2
    g : α → ↑P α
    f : α → P.M
    hyp : ∀ (x : α), Eq (f x).dest (P.map f (g x))
    x : α
    ⊢ ∀ (x : α), (fun x => True) x → Exists fun a => Exists fun f_1 => Exists fun  …
  -/
  clear x
  /-
    P : PFunctor.{u}
    α : Type u_2
    g : α → ↑P α
    f : α → P.M
    hyp : ∀ (x : α), Eq (f x).dest (P.map f (g x))
    ⊢ ∀ (x : α), (fun x => True) x → Exists fun a => Exists fun f_1 => Exists fun  …
  -/
  intro x _
  /-
    P : PFunctor.{u}
    α : Type u_2
    g : α → ↑P α
    f : α → P.M
    hyp : ∀ (x : α), Eq (f x).dest (P.map f (g x))
    x : α
    a✝ : True
    ⊢ Exists fun a => Exists fun f_1 => Exists fun f' => And (Eq (f x).dest ⟨a, f_ …
  -/
  cases' gxeq : g x with a f'
  /-
    case mk
    P : PFunctor.{u}
    α : Type u_2
    g : α → ↑P α
    f : α → P.M
    hyp : ∀ (x : α), Eq (f x).dest (P.map f (g x))
    x : α
    a✝ : True
    a : P.A
    f' : P.B a → α
    gxeq : Eq (g x) ⟨a, f'⟩
    ⊢ Exists fun a => Exists fun f_1 => Exists fun f' => And (Eq (f x).dest ⟨a, f_ …
  -/
  have h₀ : M.dest (f x) = ⟨a, f ∘ f'⟩ := by rw [hyp, gxeq, PFunctor.map_eq]
  /-
    case mk
    P : PFunctor.{u}
    α : Type u_2
    g : α → ↑P α
    f : α → P.M
    hyp : ∀ (x : α), Eq (f x).dest (P.map f (g x))
    x : α
    a✝ : True
    a : P.A
    f' : P.B a → α
    gxeq : Eq (g x) ⟨a, f'⟩
    h₀ : Eq (f x).dest ⟨a, Function.comp f f'⟩
    ⊢ Exists fun a => Exists fun f_1 => Exists fun f' => And (Eq (f x).dest ⟨a, f_ …
  -/
  have h₁ : M.dest (M.corec g x) = ⟨a, M.corec g ∘ f'⟩ := by rw [dest_corec, gxeq, PFunctor.map_eq]
  /-
    case mk
    P : PFunctor.{u}
    α : Type u_2
    g : α → ↑P α
    f : α → P.M
    hyp : ∀ (x : α), Eq (f x).dest (P.map f (g x))
    x : α
    a✝ : True
    a : P.A
    f' : P.B a → α
    gxeq : Eq (g x) ⟨a, f'⟩
    h₀ : Eq (f x).dest ⟨a, Function.comp f f'⟩
    h₁ : Eq (PFunctor.M.corec g x).dest ⟨a, Function.comp (PFunctor.M.corec g) f'⟩
    ⊢ Exists fun a => Exists fun f_1 => Exists fun f' => And (Eq (f x).dest ⟨a, f_ …
  -/
  refine ⟨_, _, _, h₀, h₁, ?_⟩
  /-
    case mk
    P : PFunctor.{u}
    α : Type u_2
    g : α → ↑P α
    f : α → P.M
    hyp : ∀ (x : α), Eq (f x).dest (P.map f (g x))
    x : α
    a✝ : True
    a : P.A
    f' : P.B a → α
    gxeq : Eq (g x) ⟨a, f'⟩
    h₀ : Eq (f x).dest ⟨a, Function.comp f f'⟩
    h₁ : Eq (PFunctor.M.corec g x).dest ⟨a, Function.comp (PFunctor.M.corec g) f'⟩
    ⊢ ∀ (i : P.B a), Exists fun x' => And ((fun x => True) x') (And (Eq (Function. …
  -/
  intro i
  /-
    case mk
    P : PFunctor.{u}
    α : Type u_2
    g : α → ↑P α
    f : α → P.M
    hyp : ∀ (x : α), Eq (f x).dest (P.map f (g x))
    x : α
    a✝ : True
    a : P.A
    f' : P.B a → α
    gxeq : Eq (g x) ⟨a, f'⟩
    h₀ : Eq (f x).dest ⟨a, Function.comp f f'⟩
    h₁ : Eq (PFunctor.M.corec g x).dest ⟨a, Function.comp (PFunctor.M.corec g) f'⟩
    i : P.B a
    ⊢ Exists fun x' => And ((fun x => True) x') (And (Eq (Function.comp f f' i) (f …
  -/
  exact ⟨f' i, trivial, rfl, rfl⟩
  /-
    🎉 no goals
  -/


/-- corecursor where the state of the computation can be sent downstream
in the form of a recursive call -/
def corec₁ {α : Type u} (F : ∀ X, (α → X) → α → P X) : α → M P :=
  M.corec (F _ id)


/-- corecursor where it is possible to return a fully formed value at any point
of the computation -/
def corec' {α : Type u} (F : ∀ {X : Type u}, (α → X) → α → M P ⊕ P X) (x : α) : M P :=
  corec₁
    (fun _ rec (a : M P ⊕ α) =>
      let y := a >>= F (rec ∘ Sum.inr)
      match y with
      | Sum.inr y => y
      | Sum.inl y => P.map (rec ∘ Sum.inl) (M.dest y))
    (@Sum.inr (M P) _ x)


