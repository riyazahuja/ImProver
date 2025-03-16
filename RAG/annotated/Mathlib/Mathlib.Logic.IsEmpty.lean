/-- `IsEmpty α` expresses that `α` is empty. -/
class IsEmpty (α : Sort*) : Prop where
  protected false : α → False


instance Empty.instIsEmpty : IsEmpty Empty :=
  ⟨Empty.elim⟩


instance PEmpty.instIsEmpty : IsEmpty PEmpty :=
  ⟨PEmpty.elim⟩


instance : IsEmpty False :=
  ⟨id⟩


instance Fin.isEmpty : IsEmpty (Fin 0) :=
  ⟨fun n ↦ Nat.not_lt_zero n.1 n.2⟩


instance Fin.isEmpty' : IsEmpty (Fin Nat.zero) :=
  Fin.isEmpty


protected theorem Function.isEmpty [IsEmpty β] (f : α → β) : IsEmpty α :=
  ⟨fun x ↦ IsEmpty.false (f x)⟩


theorem Function.Surjective.isEmpty [IsEmpty α] {f : α → β} (hf : f.Surjective) : IsEmpty β :=
  ⟨fun y ↦ let ⟨x, _⟩ := hf y; IsEmpty.false x⟩


instance {p : α → Sort*} [h : Nonempty α] [∀ x, IsEmpty (p x)] : IsEmpty (∀ x, p x) :=
  h.elim fun x ↦ Function.isEmpty <| Function.eval x


instance PProd.isEmpty_left [IsEmpty α] : IsEmpty (PProd α β) :=
  Function.isEmpty PProd.fst


instance PProd.isEmpty_right [IsEmpty β] : IsEmpty (PProd α β) :=
  Function.isEmpty PProd.snd


instance Prod.isEmpty_left {α β} [IsEmpty α] : IsEmpty (α × β) :=
  Function.isEmpty Prod.fst


instance Prod.isEmpty_right {α β} [IsEmpty β] : IsEmpty (α × β) :=
  Function.isEmpty Prod.snd


instance Quot.instIsEmpty {α : Sort*} [IsEmpty α] {r : α → α → Prop} : IsEmpty (Quot r) :=
  Function.Surjective.isEmpty Quot.exists_rep


instance Quotient.instIsEmpty {α : Sort*} [IsEmpty α] {s : Setoid α} : IsEmpty (Quotient s) :=
  Quot.instIsEmpty


instance [IsEmpty α] [IsEmpty β] : IsEmpty (α ⊕' β) :=
  ⟨fun x ↦ PSum.rec IsEmpty.false IsEmpty.false x⟩


instance instIsEmptySum {α β} [IsEmpty α] [IsEmpty β] : IsEmpty (α ⊕ β) :=
  ⟨fun x ↦ Sum.rec IsEmpty.false IsEmpty.false x⟩


/-- subtypes of an empty type are empty -/
instance [IsEmpty α] (p : α → Prop) : IsEmpty (Subtype p) :=
  ⟨fun x ↦ IsEmpty.false x.1⟩


/-- subtypes by an all-false predicate are false. -/
theorem Subtype.isEmpty_of_false {p : α → Prop} (hp : ∀ a, ¬p a) : IsEmpty (Subtype p) :=
  ⟨fun x ↦ hp _ x.2⟩


/-- subtypes by false are false. -/
instance Subtype.isEmpty_false : IsEmpty { _a : α // False } :=
  Subtype.isEmpty_of_false fun _ ↦ id


instance Sigma.isEmpty_left {α} [IsEmpty α] {E : α → Type*} : IsEmpty (Sigma E) :=
  Function.isEmpty Sigma.fst


/-- Eliminate out of a type that `IsEmpty` (without using projection notation). -/
@[elab_as_elim]
def isEmptyElim [IsEmpty α] {p : α → Sort*} (a : α) : p a :=
  (IsEmpty.false a).elim


theorem isEmpty_iff : IsEmpty α ↔ α → False :=
  ⟨@IsEmpty.false α, IsEmpty.mk⟩


universe u in
/-- Eliminate out of a type that `IsEmpty` (using projection notation). -/
@[elab_as_elim]
protected def elim {α : Sort u} (_ : IsEmpty α) {p : α → Sort*} (a : α) : p a :=
  isEmptyElim a


/-- Non-dependent version of `IsEmpty.elim`. Helpful if the elaborator cannot elaborate `h.elim a`
  correctly. -/
protected def elim' {β : Sort*} (h : IsEmpty α) (a : α) : β :=
  (h.false a).elim


protected theorem prop_iff {p : Prop} : IsEmpty p ↔ ¬p :=
  isEmpty_iff


@[simp]
theorem forall_iff {p : α → Prop} : (∀ a, p a) ↔ True :=
  iff_true_intro isEmptyElim


@[simp]
theorem exists_iff {p : α → Prop} : (∃ a, p a) ↔ False :=
  iff_false_intro fun ⟨x, _⟩ ↦ IsEmpty.false x

-- see Note [lower instance priority]

instance (priority := 100) : Subsingleton α :=
  ⟨isEmptyElim⟩


@[simp]
theorem not_nonempty_iff : ¬Nonempty α ↔ IsEmpty α :=
  ⟨fun h ↦ ⟨fun x ↦ h ⟨x⟩⟩, fun h1 h2 ↦ h2.elim h1.elim⟩


@[simp]
theorem not_isEmpty_iff : ¬IsEmpty α ↔ Nonempty α :=
  not_iff_comm.mp not_nonempty_iff


@[simp]
theorem isEmpty_Prop {p : Prop} : IsEmpty p ↔ ¬p := by
  /-
    p : Prop
    ⊢ Iff (IsEmpty p) (Not p)
  -/
  simp only [← not_nonempty_iff, nonempty_prop]
  /-
    🎉 no goals
  -/


@[simp]
theorem isEmpty_pi {π : α → Sort*} : IsEmpty (∀ a, π a) ↔ ∃ a, IsEmpty (π a) := by
  /-
    α : Sort u_1
    π : α → Sort u_4
    ⊢ Iff (IsEmpty ((a : α) → π a)) (Exists fun a => IsEmpty (π a))
  -/
  simp only [← not_nonempty_iff, Classical.nonempty_pi, not_forall]
  /-
    🎉 no goals
  -/


theorem isEmpty_fun : IsEmpty (α → β) ↔ Nonempty α ∧ IsEmpty β := by
  /-
    α : Sort u_1
    β : Sort u_2
    ⊢ Iff (IsEmpty (α → β)) (And (Nonempty α) (IsEmpty β))
  -/
  rw [isEmpty_pi, ← exists_true_iff_nonempty, ← exists_and_right, true_and]
  /-
    🎉 no goals
  -/


@[simp]
theorem nonempty_fun : Nonempty (α → β) ↔ IsEmpty α ∨ Nonempty β :=
                       /-
                         α : Sort u_1
                         β : Sort u_2
                         ⊢ Iff (Not (Nonempty (α → β))) (Not (Or (IsEmpty α) (Nonempty β)))
                       -/
  not_iff_not.mp <| by rw [not_or, not_nonempty_iff, not_nonempty_iff, isEmpty_fun, not_isEmpty_iff]
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem isEmpty_sigma {α} {E : α → Type*} : IsEmpty (Sigma E) ↔ ∀ a, IsEmpty (E a) := by
  /-
    α : Type u_5
    E : α → Type u_4
    ⊢ Iff (IsEmpty (Sigma E)) (∀ (a : α), IsEmpty (E a))
  -/
  simp only [← not_nonempty_iff, nonempty_sigma, not_exists]
  /-
    🎉 no goals
  -/


@[simp]
theorem isEmpty_psigma {α} {E : α → Sort*} : IsEmpty (PSigma E) ↔ ∀ a, IsEmpty (E a) := by
  /-
    α : Sort u_5
    E : α → Sort u_4
    ⊢ Iff (IsEmpty (PSigma E)) (∀ (a : α), IsEmpty (E a))
  -/
  simp only [← not_nonempty_iff, nonempty_psigma, not_exists]
  /-
    🎉 no goals
  -/


theorem isEmpty_subtype (p : α → Prop) : IsEmpty (Subtype p) ↔ ∀ x, ¬p x := by
  /-
    α : Sort u_1
    p : α → Prop
    ⊢ Iff (IsEmpty (Subtype p)) (∀ (x : α), Not (p x))
  -/
  simp only [← not_nonempty_iff, nonempty_subtype, not_exists]
  /-
    🎉 no goals
  -/


@[simp]
theorem isEmpty_prod {α β : Type*} : IsEmpty (α × β) ↔ IsEmpty α ∨ IsEmpty β := by
  /-
    α : Type u_4
    β : Type u_5
    ⊢ Iff (IsEmpty (Prod α β)) (Or (IsEmpty α) (IsEmpty β))
  -/
  simp only [← not_nonempty_iff, nonempty_prod, not_and_or]
  /-
    🎉 no goals
  -/


@[simp]
theorem isEmpty_pprod : IsEmpty (PProd α β) ↔ IsEmpty α ∨ IsEmpty β := by
  /-
    α : Sort u_1
    β : Sort u_2
    ⊢ Iff (IsEmpty (PProd α β)) (Or (IsEmpty α) (IsEmpty β))
  -/
  simp only [← not_nonempty_iff, nonempty_pprod, not_and_or]
  /-
    🎉 no goals
  -/


@[simp]
theorem isEmpty_sum {α β} : IsEmpty (α ⊕ β) ↔ IsEmpty α ∧ IsEmpty β := by
  /-
    α : Type u_4
    β : Type u_5
    ⊢ Iff (IsEmpty (Sum α β)) (And (IsEmpty α) (IsEmpty β))
  -/
  simp only [← not_nonempty_iff, nonempty_sum, not_or]
  /-
    🎉 no goals
  -/


@[simp]
theorem isEmpty_psum {α β} : IsEmpty (α ⊕' β) ↔ IsEmpty α ∧ IsEmpty β := by
  /-
    α : Sort u_4
    β : Sort u_5
    ⊢ Iff (IsEmpty (PSum α β)) (And (IsEmpty α) (IsEmpty β))
  -/
  simp only [← not_nonempty_iff, nonempty_psum, not_or]
  /-
    🎉 no goals
  -/


@[simp]
theorem isEmpty_ulift {α} : IsEmpty (ULift α) ↔ IsEmpty α := by
  /-
    α : Type u_4
    ⊢ Iff (IsEmpty (ULift.{u_5, u_4} α)) (IsEmpty α)
  -/
  simp only [← not_nonempty_iff, nonempty_ulift]
  /-
    🎉 no goals
  -/


@[simp]
theorem isEmpty_plift {α} : IsEmpty (PLift α) ↔ IsEmpty α := by
  /-
    α : Sort u_4
    ⊢ Iff (IsEmpty (PLift α)) (IsEmpty α)
  -/
  simp only [← not_nonempty_iff, nonempty_plift]
  /-
    🎉 no goals
  -/


theorem wellFounded_of_isEmpty {α} [IsEmpty α] (r : α → α → Prop) : WellFounded r :=
  ⟨isEmptyElim⟩


theorem isEmpty_or_nonempty : IsEmpty α ∨ Nonempty α :=
  (em <| IsEmpty α).elim Or.inl <| Or.inr ∘ not_isEmpty_iff.mp


@[simp]
theorem not_isEmpty_of_nonempty [h : Nonempty α] : ¬IsEmpty α :=
  not_isEmpty_iff.mpr h


theorem Function.extend_of_isEmpty [IsEmpty α] (f : α → β) (g : α → γ) (h : β → γ) :
    Function.extend f g h = h :=
  funext fun _ ↦ (Function.extend_apply' _ _ _) fun ⟨a, _⟩ ↦ isEmptyElim a


@[simp]
theorem leftTotal_empty [IsEmpty α] : LeftTotal R := by
  /-
    α : Type u_4
    β : Type u_5
    R : α → β → Prop
    inst✝ : IsEmpty α
    ⊢ Relator.LeftTotal R
  -/
  simp only [LeftTotal, IsEmpty.forall_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem rightTotal_empty [IsEmpty β] : RightTotal R := by
  /-
    α : Type u_4
    β : Type u_5
    R : α → β → Prop
    inst✝ : IsEmpty β
    ⊢ Relator.RightTotal R
  -/
  simp only [RightTotal, IsEmpty.forall_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem biTotal_empty [IsEmpty α] [IsEmpty β] : BiTotal R :=
  ⟨leftTotal_empty R, rightTotal_empty R⟩

