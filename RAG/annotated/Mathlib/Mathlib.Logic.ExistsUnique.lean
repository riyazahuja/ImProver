/-- For `p : α → Prop`, `ExistsUnique p` means that there exists a unique `x : α` with `p x`. -/
def ExistsUnique (p : α → Prop) := ∃ x, p x ∧ ∀ y, p y → y = x


/-- Checks to see that `xs` has only one binder. -/
def isExplicitBinderSingular (xs : TSyntax ``explicitBinders) : Bool :=
  match xs with
  | `(explicitBinders| $_:binderIdent $[: $_]?) => true
  | `(explicitBinders| ($_:binderIdent : $_)) => true
  | _ => false


open TSyntax.Compat in
/--
`∃! x : α, p x` means that there exists a unique `x` in `α` such that `p x`.
This is notation for `ExistsUnique (fun (x : α) ↦ p x)`.

This notation does not allow multiple binders like `∃! (x : α) (y : β), p x y`
as a shorthand for `∃! (x : α), ∃! (y : β), p x y` since it is liable to be misunderstood.
Often, the intended meaning is instead `∃! q : α × β, p q.1 q.2`.
-/
macro "∃!" xs:explicitBinders ", " b:term : term => do
  if !isExplicitBinderSingular xs then
    Macro.throwErrorAt xs "\
      The `ExistsUnique` notation should not be used with more than one binder.\n\
      \n\
      The reason for this is that `∃! (x : α), ∃! (y : β), p x y` has a completely different \
      meaning from `∃! q : α × β, p q.1 q.2`. \
      To prevent confusion, this notation requires that you be explicit \
      and use one with the correct interpretation."
  expandExplicitBinders ``ExistsUnique xs b


/--
Pretty-printing for `ExistsUnique`, following the same pattern as pretty printing for `Exists`.
However, it does *not* merge binders.
-/
@[app_unexpander ExistsUnique] def unexpandExistsUnique : Lean.PrettyPrinter.Unexpander
  | `($(_) fun $x:ident ↦ $b)                      => `(∃! $x:ident, $b)
  | `($(_) fun ($x:ident : $t) ↦ $b)               => `(∃! $x:ident : $t, $b)
  | _                                               => throw ()


/--
`∃! x ∈ s, p x` means `∃! x, x ∈ s ∧ p x`, which is to say that there exists a unique `x ∈ s`
such that `p x`.
Similarly, notations such as `∃! x ≤ n, p n` are supported,
using any relation defined using the `binder_predicate` command.
-/
syntax "∃! " binderIdent binderPred ", " term : term


macro_rules
  | `(∃! $x:ident $p:binderPred, $b) => `(∃! $x:ident, satisfies_binder_pred% $x $p ∧ $b)
  | `(∃! _ $p:binderPred, $b) => `(∃! x, satisfies_binder_pred% x $p ∧ $b)


theorem ExistsUnique.intro {p : α → Prop} (w : α)
    (h₁ : p w) (h₂ : ∀ y, p y → y = w) : ∃! x, p x := ⟨w, h₁, h₂⟩


theorem ExistsUnique.elim {p : α → Prop} {b : Prop}
    (h₂ : ∃! x, p x) (h₁ : ∀ x, p x → (∀ y, p y → y = x) → b) : b :=
  Exists.elim h₂ (fun w hw ↦ h₁ w (And.left hw) (And.right hw))


theorem existsUnique_of_exists_of_unique {p : α → Prop}
    (hex : ∃ x, p x) (hunique : ∀ y₁ y₂, p y₁ → p y₂ → y₁ = y₂) : ∃! x, p x :=
  Exists.elim hex (fun x px ↦ ExistsUnique.intro x px (fun y (h : p y) ↦ hunique y x h px))


@[deprecated (since := "2024-12-17")]
alias exists_unique_of_exists_of_unique := existsUnique_of_exists_of_unique


theorem ExistsUnique.exists {p : α → Prop} : (∃! x, p x) → ∃ x, p x | ⟨x, h, _⟩ => ⟨x, h⟩


theorem ExistsUnique.unique {p : α → Prop}
    (h : ∃! x, p x) {y₁ y₂ : α} (py₁ : p y₁) (py₂ : p y₂) : y₁ = y₂ :=
  let ⟨_, _, hy⟩ := h; (hy _ py₁).trans (hy _ py₂).symm

-- TODO
-- attribute [congr] forall_congr'
-- attribute [congr] exists_congr'

-- @[congr]

theorem existsUnique_congr {p q : α → Prop} (h : ∀ a, p a ↔ q a) : (∃! a, p a) ↔ ∃! a, q a :=
  exists_congr fun _ ↦ and_congr (h _) <| forall_congr' fun _ ↦ imp_congr_left (h _)


@[simp] theorem existsUnique_iff_exists [Subsingleton α] {p : α → Prop} :
    (∃! x, p x) ↔ ∃ x, p x :=
  ⟨fun h ↦ h.exists, Exists.imp fun x hx ↦ ⟨hx, fun y _ ↦ Subsingleton.elim y x⟩⟩


@[deprecated (since := "2024-12-17")] alias exists_unique_iff_exists := existsUnique_iff_exists


theorem existsUnique_const {b : Prop} (α : Sort*) [i : Nonempty α] [Subsingleton α] :
                            /-
                              b : Prop
                              α : Sort u_2
                              i : Nonempty α
                              inst✝ : Subsingleton α
                              ⊢ Iff (ExistsUnique fun x => b) b
                            -/
    (∃! _ : α, b) ↔ b := by simp
                            /-
                              🎉 no goals
                            -/


@[deprecated (since := "2024-12-17")] alias exists_unique_const := existsUnique_const


@[simp] theorem existsUnique_eq {a' : α} : ∃! a, a = a' := by
  /-
    α : Sort u_1
    a' : α
    ⊢ ExistsUnique fun a => Eq a a'
  -/
  simp only [eq_comm, ExistsUnique, and_self, forall_eq', exists_eq']
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-17")] alias exists_unique_eq := existsUnique_eq


/-- The difference with `existsUnique_eq` is that the equality is reversed. -/
@[simp] theorem existsUnique_eq' {a' : α} : ∃! a, a' = a := by
  /-
    α : Sort u_1
    a' : α
    ⊢ ExistsUnique fun a => Eq a' a
  -/
  simp only [ExistsUnique, and_self, forall_eq', exists_eq']
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-17")] alias exists_unique_eq' := existsUnique_eq'


                                                                     /-
                                                                       p q : Prop
                                                                       ⊢ Iff (ExistsUnique fun x => q) (And p q)
                                                                     -/
theorem existsUnique_prop {p q : Prop} : (∃! _ : p, q) ↔ p ∧ q := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[deprecated (since := "2024-12-17")] alias exists_unique_prop := existsUnique_prop


@[simp] theorem existsUnique_false : ¬∃! _ : α, False := fun ⟨_, h, _⟩ ↦ h


@[deprecated (since := "2024-12-17")] alias exists_unique_false := existsUnique_false


theorem existsUnique_prop_of_true {p : Prop} {q : p → Prop} (h : p) : (∃! h' : p, q h') ↔ q h :=
  @existsUnique_const (q h) p ⟨h⟩ _


@[deprecated (since := "2024-12-17")] alias exists_unique_prop_of_true := existsUnique_prop_of_true


theorem ExistsUnique.elim₂ {p : α → Sort*} [∀ x, Subsingleton (p x)]
    {q : ∀ (x) (_ : p x), Prop} {b : Prop} (h₂ : ∃! x, ∃! h : p x, q x h)
    (h₁ : ∀ (x) (h : p x), q x h → (∀ (y) (hy : p y), q y hy → y = x) → b) : b := by
  /-
    α : Sort u_1
    p : α → Sort u_2
    inst✝ : ∀ (x : α), Subsingleton (p x)
    q : (x : α) → p x → Prop
    b : Prop
    h₂ : ExistsUnique fun x => ExistsUnique fun h => q x h
    h₁ : ∀ (x : α) (h : p x), q x h → (∀ (y : α) (hy : p y), q y hy → Eq y x) → b
    ⊢ b
  -/
  simp only [existsUnique_iff_exists] at h₂
  /-
    α : Sort u_1
    p : α → Sort u_2
    inst✝ : ∀ (x : α), Subsingleton (p x)
    q : (x : α) → p x → Prop
    b : Prop
    h₁ : ∀ (x : α) (h : p x), q x h → (∀ (y : α) (hy : p y), q y hy → Eq y x) → b
    h₂ : ExistsUnique fun x => Exists fun h => q x h
    ⊢ b
  -/
  apply h₂.elim
  /-
    α : Sort u_1
    p : α → Sort u_2
    inst✝ : ∀ (x : α), Subsingleton (p x)
    q : (x : α) → p x → Prop
    b : Prop
    h₁ : ∀ (x : α) (h : p x), q x h → (∀ (y : α) (hy : p y), q y hy → Eq y x) → b
    h₂ : ExistsUnique fun x => Exists fun h => q x h
    ⊢ ∀ (x : α), (Exists fun h => q x h) → (∀ (y : α), (Exists fun h => q y h) → E …
  -/
  exact fun x ⟨hxp, hxq⟩ H ↦ h₁ x hxp hxq fun y hyp hyq ↦ H y ⟨hyp, hyq⟩
  /-
    🎉 no goals
  -/


theorem ExistsUnique.intro₂ {p : α → Sort*} [∀ x, Subsingleton (p x)]
    {q : ∀ (x : α) (_ : p x), Prop} (w : α) (hp : p w) (hq : q w hp)
    (H : ∀ (y) (hy : p y), q y hy → y = w) : ∃! x, ∃! hx : p x, q x hx := by
  /-
    α : Sort u_1
    p : α → Sort u_2
    inst✝ : ∀ (x : α), Subsingleton (p x)
    q : (x : α) → p x → Prop
    w : α
    hp : p w
    hq : q w hp
    H : ∀ (y : α) (hy : p y), q y hy → Eq y w
    ⊢ ExistsUnique fun x => ExistsUnique fun hx => q x hx
  -/
  simp only [existsUnique_iff_exists]
  /-
    α : Sort u_1
    p : α → Sort u_2
    inst✝ : ∀ (x : α), Subsingleton (p x)
    q : (x : α) → p x → Prop
    w : α
    hp : p w
    hq : q w hp
    H : ∀ (y : α) (hy : p y), q y hy → Eq y w
    ⊢ ExistsUnique fun x => Exists fun hx => q x hx
  -/
  exact ExistsUnique.intro w ⟨hp, hq⟩ fun y ⟨hyp, hyq⟩ ↦ H y hyp hyq
  /-
    🎉 no goals
  -/


theorem ExistsUnique.exists₂ {p : α → Sort*} {q : ∀ (x : α) (_ : p x), Prop}
    (h : ∃! x, ∃! hx : p x, q x hx) : ∃ (x : _) (hx : p x), q x hx :=
  h.exists.imp fun _ hx ↦ hx.exists


theorem ExistsUnique.unique₂ {p : α → Sort*} [∀ x, Subsingleton (p x)]
    {q : ∀ (x : α) (_ : p x), Prop} (h : ∃! x, ∃! hx : p x, q x hx) {y₁ y₂ : α}
    (hpy₁ : p y₁) (hqy₁ : q y₁ hpy₁) (hpy₂ : p y₂) (hqy₂ : q y₂ hpy₂) : y₁ = y₂ := by
  /-
    α : Sort u_1
    p : α → Sort u_2
    inst✝ : ∀ (x : α), Subsingleton (p x)
    q : (x : α) → p x → Prop
    h : ExistsUnique fun x => ExistsUnique fun hx => q x hx
    y₁ y₂ : α
    hpy₁ : p y₁
    hqy₁ : q y₁ hpy₁
    hpy₂ : p y₂
    hqy₂ : q y₂ hpy₂
    ⊢ Eq y₁ y₂
  -/
  simp only [existsUnique_iff_exists] at h
  /-
    α : Sort u_1
    p : α → Sort u_2
    inst✝ : ∀ (x : α), Subsingleton (p x)
    q : (x : α) → p x → Prop
    y₁ y₂ : α
    hpy₁ : p y₁
    hqy₁ : q y₁ hpy₁
    hpy₂ : p y₂
    hqy₂ : q y₂ hpy₂
    h : ExistsUnique fun x => Exists fun hx => q x hx
    ⊢ Eq y₁ y₂
  -/
  exact h.unique ⟨hpy₁, hqy₁⟩ ⟨hpy₂, hqy₂⟩
  /-
    🎉 no goals
  -/

