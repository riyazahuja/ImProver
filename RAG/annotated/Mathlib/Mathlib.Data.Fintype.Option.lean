instance {α : Type*} [Fintype α] : Fintype (Option α) :=
                                       /-
                                         α✝ : Type u_1
                                         β : Type u_2
                                         α : Type u_3
                                         inst✝ : Fintype α
                                         a : Option α
                                         ⊢ Membership.mem (Finset.insertNone Finset.univ) a
                                       -/
  ⟨Finset.insertNone univ, fun a => by simp⟩
                                       /-
                                         🎉 no goals
                                       -/


theorem univ_option (α : Type*) [Fintype α] : (univ : Finset (Option α)) = insertNone univ :=
  rfl


@[simp]
theorem Fintype.card_option {α : Type*} [Fintype α] :
    Fintype.card (Option α) = Fintype.card α + 1 :=
                        /-
                          α : Type u_3
                          inst✝ : Fintype α
                          ⊢ Not (Membership.mem (Finset.map Function.Embedding.some Finset.univ) Option. …
                        -/
  (Finset.card_cons (by simp)).trans <| congr_arg₂ _ (card_map _) rfl
                        /-
                          🎉 no goals
                        -/


/-- If `Option α` is a `Fintype` then so is `α` -/
def fintypeOfOption {α : Type*} [Fintype (Option α)] : Fintype α :=
  ⟨Finset.eraseNone (Fintype.elems (α := Option α)), fun x =>
    mem_eraseNone.mpr (Fintype.complete (some x))⟩


/-- A type is a `Fintype` if its successor (using `Option`) is a `Fintype`. -/
def fintypeOfOptionEquiv [Fintype α] (f : α ≃ Option β) : Fintype β :=
  haveI := Fintype.ofEquiv _ f
  fintypeOfOption


/-- A recursor principle for finite types, analogous to `Nat.rec`. It effectively says
that every `Fintype` is either `Empty` or `Option α`, up to an `Equiv`. -/
def truncRecEmptyOption {P : Type u → Sort v} (of_equiv : ∀ {α β}, α ≃ β → P α → P β)
    (h_empty : P PEmpty) (h_option : ∀ {α} [Fintype α] [DecidableEq α], P α → P (Option α))
    (α : Type u) [Fintype α] [DecidableEq α] : Trunc (P α) := by
  suffices ∀ n : ℕ, Trunc (P (ULift <| Fin n)) by
    apply Trunc.bind (this (Fintype.card α))
    intro h
    apply Trunc.map _ (Fintype.truncEquivFin α)
    intro e
    exact of_equiv (Equiv.ulift.trans e.symm) h
  /-
    α✝ : Type u_1
    β : Type u_2
    P : Type u → Sort v
    of_equiv : {α β : Type u} → Equiv α β → P α → P β
    h_empty : P PEmpty.{u + 1}
    h_option : {α : Type u} → [inst : Fintype α] → [inst : DecidableEq α] → P α →  …
    α : Type u
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    ⊢ (n : Nat) → Trunc (P (ULift.{u, 0} (Fin n)))
  -/
  apply ind where
  /-
    🎉 no goals
  -/
    -- Porting note: do a manual recursion, instead of `induction` tactic,
    -- to ensure the result is computable
    /-- Internal induction hypothesis -/
    ind : ∀ n : ℕ, Trunc (P (ULift <| Fin n))
    | Nat.zero => by
          /-
            α✝ : Type u_1
            β : Type u_2
            P : Type u → Sort v
            of_equiv : {α β : Type u} → Equiv α β → P α → P β
            h_empty : P PEmpty.{u + 1}
            h_option : {α : Type u} → [inst : Fintype α] → [inst : DecidableEq α] → P α →  …
            α : Type u
            inst✝¹ : Fintype α
            inst✝ : DecidableEq α
            this : Eq (Fintype.card PEmpty.{?u.2445 + 1}) (Fintype.card (ULift.{?u.2455, 0 …
            ⊢ Trunc (P (ULift.{u, 0} (Fin Nat.zero)))
          -/
          have : card PEmpty = card (ULift (Fin 0)) := by simp only [card_fin, card_pempty,
          /-
            α✝ : Type u_1
            β : Type u_2
            P : Type u → Sort v
            of_equiv : {α β : Type u} → Equiv α β → P α → P β
            h_empty : P PEmpty.{u + 1}
            h_option : {α : Type u} → [inst : Fintype α] → [inst : DecidableEq α] → P α →  …
            α : Type u
            inst✝¹ : Fintype α
            inst✝ : DecidableEq α
            this : Eq (Fintype.card PEmpty.{?u.2445 + 1}) (Fintype.card (ULift.{?u.2455, 0 …
            ⊢ Equiv PEmpty.{?u.2445 + 1} (ULift.{?u.2455, 0} (Fin 0)) → Trunc (P (ULift.{u …
          -/
                                                                     card_ulift]
          /-
            α✝ : Type u_1
            β : Type u_2
            P : Type u → Sort v
            of_equiv : {α β : Type u} → Equiv α β → P α → P β
            h_empty : P PEmpty.{u + 1}
            h_option : {α : Type u} → [inst : Fintype α] → [inst : DecidableEq α] → P α →  …
            α : Type u
            inst✝¹ : Fintype α
            inst✝ : DecidableEq α
            this : Eq (Fintype.card PEmpty.{?u.2445 + 1}) (Fintype.card (ULift.{?u.2455, 0 …
            e : Equiv PEmpty.{?u.2445 + 1} (ULift.{?u.2455, 0} (Fin 0))
            ⊢ Trunc (P (ULift.{u, 0} (Fin Nat.zero)))
          -/
          apply Trunc.bind (truncEquivOfCardEq this)
          /-
            case a
            α✝ : Type u_1
            β : Type u_2
            P : Type u → Sort v
            of_equiv : {α β : Type u} → Equiv α β → P α → P β
            h_empty : P PEmpty.{u + 1}
            h_option : {α : Type u} → [inst : Fintype α] → [inst : DecidableEq α] → P α →  …
            α : Type u
            inst✝¹ : Fintype α
            inst✝ : DecidableEq α
            this : Eq (Fintype.card PEmpty.{?u.2445 + 1}) (Fintype.card (ULift.{?u.2455, 0 …
            e : Equiv PEmpty.{?u.2445 + 1} (ULift.{?u.2455, 0} (Fin 0))
            ⊢ P (ULift.{u, 0} (Fin Nat.zero))
          -/
          intro e
          /-
            🎉 no goals
          -/
          apply Trunc.mk
          exact of_equiv e h_empty
      | Nat.succ n => by
          /-
            α✝ : Type u_1
            β : Type u_2
            P : Type u → Sort v
            of_equiv : {α β : Type u} → Equiv α β → P α → P β
            h_empty : P PEmpty.{u + 1}
            h_option : {α : Type u} → [inst : Fintype α] → [inst : DecidableEq α] → P α →  …
            α : Type u
            inst✝¹ : Fintype α
            inst✝ : DecidableEq α
            n : Nat
            this : Eq (Fintype.card (Option (ULift.{?u.2974, 0} (Fin n)))) (Fintype.card ( …
            ⊢ Trunc (P (ULift.{u, 0} (Fin n.succ)))
          -/
          have : card (Option (ULift (Fin n))) = card (ULift (Fin n.succ)) := by
          /-
            α✝ : Type u_1
            β : Type u_2
            P : Type u → Sort v
            of_equiv : {α β : Type u} → Equiv α β → P α → P β
            h_empty : P PEmpty.{u + 1}
            h_option : {α : Type u} → [inst : Fintype α] → [inst : DecidableEq α] → P α →  …
            α : Type u
            inst✝¹ : Fintype α
            inst✝ : DecidableEq α
            n : Nat
            this : Eq (Fintype.card (Option (ULift.{?u.2974, 0} (Fin n)))) (Fintype.card ( …
            ⊢ Equiv (Option (ULift.{?u.2974, 0} (Fin n))) (ULift.{?u.2995, 0} (Fin n.succ) …
          -/
            simp only [card_fin, card_option, card_ulift]
          /-
            α✝ : Type u_1
            β : Type u_2
            P : Type u → Sort v
            of_equiv : {α β : Type u} → Equiv α β → P α → P β
            h_empty : P PEmpty.{u + 1}
            h_option : {α : Type u} → [inst : Fintype α] → [inst : DecidableEq α] → P α →  …
            α : Type u
            inst✝¹ : Fintype α
            inst✝ : DecidableEq α
            n : Nat
            this : Eq (Fintype.card (Option (ULift.{?u.2974, 0} (Fin n)))) (Fintype.card ( …
            e : Equiv (Option (ULift.{?u.2974, 0} (Fin n))) (ULift.{?u.2995, 0} (Fin n.suc …
            ⊢ Trunc (P (ULift.{u, 0} (Fin n.succ)))
          -/
          apply Trunc.bind (truncEquivOfCardEq this)
          /-
            α✝ : Type u_1
            β : Type u_2
            P : Type u → Sort v
            of_equiv : {α β : Type u} → Equiv α β → P α → P β
            h_empty : P PEmpty.{u + 1}
            h_option : {α : Type u} → [inst : Fintype α] → [inst : DecidableEq α] → P α →  …
            α : Type u
            inst✝¹ : Fintype α
            inst✝ : DecidableEq α
            n : Nat
            this : Eq (Fintype.card (Option (ULift.{?u.2974, 0} (Fin n)))) (Fintype.card ( …
            e : Equiv (Option (ULift.{?u.2974, 0} (Fin n))) (ULift.{?u.2995, 0} (Fin n.suc …
            ⊢ P (ULift.{u, 0} (Fin n)) → P (ULift.{u, 0} (Fin n.succ))
          -/
          intro e
          /-
            α✝ : Type u_1
            β : Type u_2
            P : Type u → Sort v
            of_equiv : {α β : Type u} → Equiv α β → P α → P β
            h_empty : P PEmpty.{u + 1}
            h_option : {α : Type u} → [inst : Fintype α] → [inst : DecidableEq α] → P α →  …
            α : Type u
            inst✝¹ : Fintype α
            inst✝ : DecidableEq α
            n : Nat
            this : Eq (Fintype.card (Option (ULift.{?u.2974, 0} (Fin n)))) (Fintype.card ( …
            e : Equiv (Option (ULift.{?u.2974, 0} (Fin n))) (ULift.{?u.2995, 0} (Fin n.suc …
            ih : P (ULift.{u, 0} (Fin n))
            ⊢ P (ULift.{u, 0} (Fin n.succ))
          -/
          apply Trunc.map _ (ind n)
          /-
            🎉 no goals
          -/
          intro ih
          exact of_equiv e (h_option ih)

-- Porting note: due to instance inference issues in `SetTheory.Cardinal.Basic`
-- I had to explicitly name `h_fintype` in order to access it manually.
-- was `[Fintype α]`

/-- An induction principle for finite types, analogous to `Nat.rec`. It effectively says
that every `Fintype` is either `Empty` or `Option α`, up to an `Equiv`. -/
@[elab_as_elim]
theorem induction_empty_option {P : ∀ (α : Type u) [Fintype α], Prop}
    (of_equiv : ∀ (α β) [Fintype β] (e : α ≃ β), @P α (@Fintype.ofEquiv α β ‹_› e.symm) → @P β ‹_›)
    (h_empty : P PEmpty) (h_option : ∀ (α) [Fintype α], P α → P (Option α)) (α : Type u)
    [h_fintype : Fintype α] : P α := by
  obtain ⟨p⟩ :=
    let f_empty := fun i => by convert h_empty
    let h_option : ∀ {α : Type u} [Fintype α] [DecidableEq α],
          (∀ (h : Fintype α), P α) → ∀ (h : Fintype (Option α)), P (Option α)  := by
      rintro α hα - Pα hα'
      convert h_option α (Pα _)
    @truncRecEmptyOption (fun α => ∀ h, @P α h) (@fun α β e hα hβ => @of_equiv α β hβ e (hα _))
      f_empty h_option α _ (Classical.decEq α)
  /-
    case mk
    P : (α : Type u) → [inst : Fintype α] → Prop
    of_equiv : ∀ (α β : Type u) [inst : Fintype β] (e : Equiv α β), P α → P β
    h_empty : P PEmpty.{u + 1}
    h_option : ∀ (α : Type u) [inst : Fintype α], P α → P (Option α)
    α : Type u
    h_fintype : Fintype α
    x✝ : Trunc (∀ (h : Fintype α), P α)
    p : ∀ (h : Fintype α), P α
    ⊢ P α
  -/
  exact p _
  /-
    🎉 no goals
  -/
  -- ·


/-- An induction principle for finite types, analogous to `Nat.rec`. It effectively says
that every `Fintype` is either `Empty` or `Option α`, up to an `Equiv`. -/
theorem Finite.induction_empty_option {P : Type u → Prop} (of_equiv : ∀ {α β}, α ≃ β → P α → P β)
    (h_empty : P PEmpty) (h_option : ∀ {α} [Fintype α], P α → P (Option α)) (α : Type u)
    [Finite α] : P α := by
  /-
    P : Type u → Prop
    of_equiv : ∀ {α β : Type u}, Equiv α β → P α → P β
    h_empty : P PEmpty.{u + 1}
    h_option : ∀ {α : Type u} [inst : Fintype α], P α → P (Option α)
    α : Type u
    inst✝ : Finite α
    ⊢ P α
  -/
  cases nonempty_fintype α
  /-
    case intro
    P : Type u → Prop
    of_equiv : ∀ {α β : Type u}, Equiv α β → P α → P β
    h_empty : P PEmpty.{u + 1}
    h_option : ∀ {α : Type u} [inst : Fintype α], P α → P (Option α)
    α : Type u
    inst✝ : Finite α
    val✝ : Fintype α
    ⊢ P α
  -/
  refine Fintype.induction_empty_option ?_ ?_ ?_ α
  /-
    case intro.refine_1
    P : Type u → Prop
    of_equiv : ∀ {α β : Type u}, Equiv α β → P α → P β
    h_empty : P PEmpty.{u + 1}
    h_option : ∀ {α : Type u} [inst : Fintype α], P α → P (Option α)
    α : Type u
    inst✝ : Finite α
    val✝ : Fintype α
    ⊢ ∀ (α β : Type u) [inst : Fintype β], Equiv α β → P α → P β
  -/
  exacts [fun α β _ => of_equiv, h_empty, @h_option]
  /-
    🎉 no goals
  -/

