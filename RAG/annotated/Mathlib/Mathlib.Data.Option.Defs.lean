/-- Traverse an object of `Option α` with a function `f : α → F β` for an applicative `F`. -/
protected def traverse.{u, v}
    {F : Type u → Type v} [Applicative F] {α : Type*} {β : Type u} (f : α → F β) :
    Option α → F (Option β)
  | none => pure none
  | some x => some <$> f x


/-- An elimination principle for `Option`. It is a nondependent version of `Option.rec`. -/
protected def elim' (b : β) (f : α → β) : Option α → β
  | some a => f a
  | none => b


@[simp]
theorem elim'_none (b : β) (f : α → β) : Option.elim' b f none = b := rfl

@[simp]
theorem elim'_some {a : α} (b : β) (f : α → β) : Option.elim' b f (some a) = f a := rfl

-- Porting note: this lemma was introduced because it is necessary
-- in `CategoryTheory.Category.PartialFun`

lemma elim'_eq_elim {α β : Type*} (b : β) (f : α → β) (a : Option α) :
    Option.elim' b f a = Option.elim a b f := by
  /-
    α : Type u_3
    β : Type u_4
    b : β
    f : α → β
    a : Option α
    ⊢ Eq (Option.elim' b f a) (a.elim b f)
  -/
              /-
                🎉 no goals
              -/
  cases a <;> rfl
              /-
                🎉 no goals
              -/



                                                                      /-
                                                                        α : Type u_3
                                                                        a b : α
                                                                        ⊢ Iff (Membership.mem (Option.some b) a) (Eq b a)
                                                                      -/
theorem mem_some_iff {α : Type*} {a b : α} : a ∈ some b ↔ b = a := by simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- `o = none` is decidable even if the wrapped type does not have decidable equality.
This is not an instance because it is not definitionally equal to `Option.decidableEq`.
Try to use `o.isNone` or `o.isSome` instead.
-/
@[inline]
def decidableEqNone {o : Option α} : Decidable (o = none) :=
  decidable_of_decidable_of_iff isNone_iff_eq_none


instance decidableForallMem {p : α → Prop} [DecidablePred p] :
    ∀ o : Option α, Decidable (∀ a ∈ o, p a)
                       /-
                         α : Type u_1
                         β : Type u_2
                         p : α → Prop
                         inst✝ : DecidablePred p
                         ⊢ ∀ (a : α), Membership.mem Option.none a → p a
                       -/
  | none => isTrue (by simp [false_imp_iff])
                       /-
                         🎉 no goals
                       -/
  | some a =>
      if h : p a then isTrue fun _ e ↦ some_inj.1 e ▸ h
      else isFalse <| mt (fun H ↦ H _ rfl) h


instance decidableExistsMem {p : α → Prop} [DecidablePred p] :
    ∀ o : Option α, Decidable (∃ a ∈ o, p a)
                                         /-
                                           α : Type u_1
                                           β : Type u_2
                                           p : α → Prop
                                           inst✝ : DecidablePred p
                                           x✝ : Exists fun a => And (Membership.mem Option.none a) (p a)
                                           a : α
                                           h : Membership.mem Option.none a
                                           right✝ : p a
                                           ⊢ False
                                         -/
  | none => isFalse fun ⟨a, ⟨h, _⟩⟩ ↦ by cases h
                                         /-
                                           🎉 no goals
                                         -/
  | some a => if h : p a then isTrue <| ⟨_, rfl, h⟩ else isFalse fun ⟨_, ⟨rfl, hn⟩⟩ ↦ h hn


/-- Inhabited `get` function. Returns `a` if the input is `some a`, otherwise returns `default`. -/
abbrev iget [Inhabited α] : Option α → α
  | some x => x
  | none => default


theorem iget_some [Inhabited α] {a : α} : (some a).iget = a :=
  rfl


instance liftOrGet_isCommutative (f : α → α → α) [Std.Commutative f] :
    Std.Commutative (liftOrGet f) :=
                /-
                  α : Type u_1
                  β : Type u_2
                  f : α → α → α
                  inst✝ : Std.Commutative f
                  a b : Option α
                  ⊢ Eq (Option.liftOrGet f a b) (Option.liftOrGet f b a)
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
  ⟨fun a b ↦ by cases a <;> cases b <;> simp [liftOrGet, Std.Commutative.comm]⟩
                                        /-
                                          🎉 no goals
                                        -/


instance liftOrGet_isAssociative (f : α → α → α) [Std.Associative f] :
    Std.Associative (liftOrGet f) :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    f : α → α → α
                    inst✝ : Std.Associative f
                    a b c : Option α
                    ⊢ Eq (Option.liftOrGet f (Option.liftOrGet f a b) c) (Option.liftOrGet f a (Op …
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
                                                      /-
                                                        🎉 no goals
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
  ⟨fun a b c ↦ by cases a <;> cases b <;> cases c <;> simp [liftOrGet, Std.Associative.assoc]⟩
                                                      /-
                                                        🎉 no goals
                                                      -/


instance liftOrGet_isIdempotent (f : α → α → α) [Std.IdempotentOp f] :
    Std.IdempotentOp (liftOrGet f) :=
              /-
                α : Type u_1
                β : Type u_2
                f : α → α → α
                inst✝ : Std.IdempotentOp f
                a : Option α
                ⊢ Eq (Option.liftOrGet f a a) a
              -/
                          /-
                            🎉 no goals
                          -/
  ⟨fun a ↦ by cases a <;> simp [liftOrGet, Std.IdempotentOp.idempotent]⟩
                          /-
                            🎉 no goals
                          -/


instance liftOrGet_isId (f : α → α → α) : Std.LawfulIdentity (liftOrGet f) none where
                  /-
                    α : Type u_1
                    β : Type u_2
                    f : α → α → α
                    a : Option α
                    ⊢ Eq (Option.liftOrGet f Option.none a) a
                  -/
                              /-
                                🎉 no goals
                              -/
  left_id a := by cases a <;> simp [liftOrGet]
                              /-
                                🎉 no goals
                              -/
                   /-
                     α : Type u_1
                     β : Type u_2
                     f : α → α → α
                     a : Option α
                     ⊢ Eq (Option.liftOrGet f a Option.none) a
                   -/
                               /-
                                 🎉 no goals
                               -/
  right_id a := by cases a <;> simp [liftOrGet]
                               /-
                                 🎉 no goals
                               -/


/-- Convert `undef` to `none` to make an `LOption` into an `Option`. -/
def _root_.Lean.LOption.toOption {α} : Lean.LOption α → Option α
  | .some a => some a
  | _ => none


