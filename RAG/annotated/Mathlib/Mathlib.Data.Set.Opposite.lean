/-- The opposite of a set `s` is the set obtained by taking the opposite of each member of `s`. -/
protected def op (s : Set α) : Set αᵒᵖ :=
  unop ⁻¹' s


/-- The unop of a set `s` is the set obtained by taking the unop of each member of `s`. -/
protected def unop (s : Set αᵒᵖ) : Set α :=
  op ⁻¹' s


@[simp]
theorem mem_op {s : Set α} {a : αᵒᵖ} : a ∈ s.op ↔ unop a ∈ s :=
  Iff.rfl


@[simp 1100]
                                                                  /-
                                                                    α : Type u_1
                                                                    s : Set α
                                                                    a : α
                                                                    ⊢ Iff (Membership.mem s.op { unop := a }) (Membership.mem s a)
                                                                  -/
theorem op_mem_op {s : Set α} {a : α} : op a ∈ s.op ↔ a ∈ s := by rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem mem_unop {s : Set αᵒᵖ} {a : α} : a ∈ s.unop ↔ op a ∈ s :=
  Iff.rfl


@[simp 1100]
                                                                              /-
                                                                                α : Type u_1
                                                                                s : Set (Opposite α)
                                                                                a : Opposite α
                                                                                ⊢ Iff (Membership.mem s.unop (Opposite.unop a)) (Membership.mem s a)
                                                                              -/
theorem unop_mem_unop {s : Set αᵒᵖ} {a : αᵒᵖ} : unop a ∈ s.unop ↔ a ∈ s := by rfl
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp]
theorem op_unop (s : Set α) : s.op.unop = s := rfl


@[simp]
theorem unop_op (s : Set αᵒᵖ) : s.unop.op = s := rfl


/-- The members of the opposite of a set are in bijection with the members of the set itself. -/
@[simps]
def opEquiv_self (s : Set α) : s.op ≃ s :=
  ⟨fun x ↦ ⟨unop x, x.2⟩, fun x ↦ ⟨op x, x.2⟩, fun _ ↦ rfl, fun _ ↦ rfl⟩


/-- Taking opposites as an equivalence of powersets. -/
@[simps]
def opEquiv : Set α ≃ Set αᵒᵖ :=
  ⟨Set.op, Set.unop, op_unop, unop_op⟩


@[simp]
theorem singleton_op (x : α) : ({x} : Set α).op = {op x} := by
  /-
    α : Type u_1
    x : α
    ⊢ Eq (Singleton.singleton x).op (Singleton.singleton { unop := x })
  -/
  ext
  /-
    case h
    α : Type u_1
    x : α
    x✝ : Opposite α
    ⊢ Iff (Membership.mem (Singleton.singleton x).op x✝) (Membership.mem (Singleto …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      x : α
      x✝ : Opposite α
      ⊢ Membership.mem (Singleton.singleton x).op x✝ → Membership.mem (Singleton.sin …
    -/
  · apply unop_injective
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      x : α
      x✝ : Opposite α
      ⊢ Membership.mem (Singleton.singleton { unop := x }) x✝ → Membership.mem (Sing …
    -/
  · apply op_injective
    /-
      🎉 no goals
    -/


@[simp]
theorem singleton_unop (x : αᵒᵖ) : ({x} : Set αᵒᵖ).unop = {unop x} := by
  /-
    α : Type u_1
    x : Opposite α
    ⊢ Eq (Singleton.singleton x).unop (Singleton.singleton (Opposite.unop x))
  -/
  ext
  /-
    case h
    α : Type u_1
    x : Opposite α
    x✝ : α
    ⊢ Iff (Membership.mem (Singleton.singleton x).unop x✝) (Membership.mem (Single …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      x : Opposite α
      x✝ : α
      ⊢ Membership.mem (Singleton.singleton x).unop x✝ → Membership.mem (Singleton.s …
    -/
  · apply op_injective
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      x : Opposite α
      x✝ : α
      ⊢ Membership.mem (Singleton.singleton (Opposite.unop x)) x✝ → Membership.mem ( …
    -/
  · apply unop_injective
    /-
      🎉 no goals
    -/


@[simp 1100]
theorem singleton_op_unop (x : α) : ({op x} : Set αᵒᵖ).unop = {x} := by
  /-
    α : Type u_1
    x : α
    ⊢ Eq (Singleton.singleton { unop := x }).unop (Singleton.singleton x)
  -/
  ext
  /-
    case h
    α : Type u_1
    x x✝ : α
    ⊢ Iff (Membership.mem (Singleton.singleton { unop := x }).unop x✝) (Membership …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      x x✝ : α
      ⊢ Membership.mem (Singleton.singleton { unop := x }).unop x✝ → Membership.mem  …
    -/
  · apply op_injective
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      x x✝ : α
      ⊢ Membership.mem (Singleton.singleton x) x✝ → Membership.mem (Singleton.single …
    -/
  · apply unop_injective
    /-
      🎉 no goals
    -/


@[simp 1100]
theorem singleton_unop_op (x : αᵒᵖ) : ({unop x} : Set α).op = {x} := by
  /-
    α : Type u_1
    x : Opposite α
    ⊢ Eq (Singleton.singleton (Opposite.unop x)).op (Singleton.singleton x)
  -/
  ext
  /-
    case h
    α : Type u_1
    x x✝ : Opposite α
    ⊢ Iff (Membership.mem (Singleton.singleton (Opposite.unop x)).op x✝) (Membersh …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      x x✝ : Opposite α
      ⊢ Membership.mem (Singleton.singleton (Opposite.unop x)).op x✝ → Membership.me …
    -/
  · apply unop_injective
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      x x✝ : Opposite α
      ⊢ Membership.mem (Singleton.singleton x) x✝ → Membership.mem (Singleton.single …
    -/
  · apply op_injective
    /-
      🎉 no goals
    -/


