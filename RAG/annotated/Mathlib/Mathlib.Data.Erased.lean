/-- `Erased α` is the same as `α`, except that the elements
  of `Erased α` are erased in the VM in the same way as types
  and proofs. This can be used to track data without storing it
  literally. -/
def Erased (α : Sort u) : Sort max 1 u :=
  Σ's : α → Prop, ∃ a, (fun b => a = b) = s


/-- Erase a value. -/
@[inline]
def mk {α} (a : α) : Erased α :=
  ⟨fun b => a = b, a, rfl⟩


/-- Extracts the erased value, noncomputably. -/
noncomputable def out {α} : Erased α → α
  | ⟨_, h⟩ => Classical.choose h


/-- Extracts the erased value, if it is a type.

Note: `(mk a).OutType` is not definitionally equal to `a`.
-/
abbrev OutType (a : Erased (Sort u)) : Sort u :=
  out a


/-- Extracts the erased value, if it is a proof. -/
theorem out_proof {p : Prop} (a : Erased p) : p :=
  out a


@[simp]
theorem out_mk {α} (a : α) : (mk a).out = a := by
  /-
    α : Sort u_1
    a : α
    ⊢ Eq (Erased.mk a).out a
  -/
  let h := (mk a).2; show Classical.choose h = a
  /-
    α : Sort u_1
    a : α
    h : Exists fun a_1 => Eq (fun b => Eq a_1 b) (Erased.mk a).fst := (Erased.mk a …
    ⊢ Eq (Classical.choose h) a
  -/
  have := Classical.choose_spec h
  /-
    α : Sort u_1
    a : α
    h : Exists fun a_1 => Eq (fun b => Eq a_1 b) (Erased.mk a).fst := (Erased.mk a …
    this : Eq (fun b => Eq (Classical.choose h) b) (Erased.mk a).fst
    ⊢ Eq (Classical.choose h) a
  -/
  exact cast (congr_fun this a).symm rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_out {α} : ∀ a : Erased α, mk (out a) = a
                 /-
                   α : Sort u_1
                   s : α → Prop
                   h : Exists fun a => Eq (fun b => Eq a b) s
                   ⊢ Eq (Erased.mk (Erased.out ⟨s, h⟩)) ⟨s, h⟩
                 -/
  | ⟨s, h⟩ => by simp only [mk]; congr; exact Classical.choose_spec h
                                        /-
                                          🎉 no goals
                                        -/


@[ext]
                                                                       /-
                                                                         α : Sort u_1
                                                                         a b : Erased α
                                                                         h : Eq a.out b.out
                                                                         ⊢ Eq a b
                                                                       -/
theorem out_inj {α} (a b : Erased α) (h : a.out = b.out) : a = b := by simpa using congr_arg mk h
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- Equivalence between `Erased α` and `α`. -/
noncomputable def equiv (α) : Erased α ≃ α :=
  ⟨out, mk, mk_out, out_mk⟩


instance (α : Type u) : Repr (Erased α) :=
  ⟨fun _ _ => "Erased"⟩


instance (α : Type u) : ToString (Erased α) :=
  ⟨fun _ => "Erased"⟩

-- Porting note: Deleted `has_to_format`


/-- Computably produce an erased value from a proof of nonemptiness. -/
def choice {α} (h : Nonempty α) : Erased α :=
  mk (Classical.choice h)


@[simp]
theorem nonempty_iff {α} : Nonempty (Erased α) ↔ Nonempty α :=
  ⟨fun ⟨a⟩ => ⟨a.out⟩, fun ⟨a⟩ => ⟨mk a⟩⟩


instance {α} [h : Nonempty α] : Inhabited (Erased α) :=
  ⟨choice h⟩


/-- `(>>=)` operation on `Erased`.

This is a separate definition because `α` and `β` can live in different
universes (the universe is fixed in `Monad`).
-/
def bind {α β} (a : Erased α) (f : α → Erased β) : Erased β :=
  ⟨fun b => (f a.out).1 b, (f a.out).2⟩


@[simp]
theorem bind_eq_out {α β} (a f) : @bind α β a f = f a.out := rfl


/-- Collapses two levels of erasure.
-/
def join {α} (a : Erased (Erased α)) : Erased α :=
  bind a id


@[simp]
theorem join_eq_out {α} (a) : @join α a = a.out :=
  bind_eq_out _ _


/-- `(<$>)` operation on `Erased`.

This is a separate definition because `α` and `β` can live in different
universes (the universe is fixed in `Functor`).
-/
def map {α β} (f : α → β) (a : Erased α) : Erased β :=
  bind a (mk ∘ f)


@[simp]
                                                                                 /-
                                                                                   α : Sort u_1
                                                                                   β : Sort u_2
                                                                                   f : α → β
                                                                                   a : Erased α
                                                                                   ⊢ Eq (Erased.map f a).out (f a.out)
                                                                                 -/
theorem map_out {α β} {f : α → β} (a : Erased α) : (a.map f).out = f a.out := by simp [map]
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


protected instance Monad : Monad Erased where
  pure := @mk
  bind := @bind
  map := @map


@[simp]
theorem pure_def {α} : (pure : α → Erased α) = @mk _ :=
  rfl


@[simp]
theorem bind_def {α β} : ((· >>= ·) : Erased α → (α → Erased β) → Erased β) = @bind _ _ :=
  rfl


@[simp]
theorem map_def {α β} : ((· <$> ·) : (α → β) → Erased α → Erased β) = @map _ _ :=
  rfl

-- Porting note: Old proof `by refine' { .. } <;> intros <;> ext <;> simp`

protected instance instLawfulMonad : LawfulMonad Erased :=
                 /-
                   ⊢ ∀ {α : Type u_1} (x : Erased α), Eq (Functor.map id x) x
                 -/
                    /-
                      ⊢ ∀ {α β : Type u_1}, Eq Functor.mapConst (Function.comp Functor.map (Function …
                    -/
  { id_map := by intros; ext; simp
                                 /-
                                   🎉 no goals
                                 -/
                              /-
                                🎉 no goals
                              -/
    map_const := by intros; ext; simp [Functor.mapConst]
                    /-
                      ⊢ ∀ {α β : Type u_1} (x : α) (f : α → Erased β), Eq (Bind.bind (Pure.pure x) f …
                    -/
    pure_bind := by intros; ext; simp
                         /-
                           ⊢ ∀ {α β : Type u_1} (f : α → β) (x : Erased α), Eq (Bind.bind x fun a => Pure …
                         -/
                                 /-
                                   🎉 no goals
                                 -/
                     /-
                       ⊢ ∀ {α β : Type u_1} (x : Erased α) (y : Erased β), Eq (SeqLeft.seqLeft x fun  …
                     -/
                                      /-
                                        🎉 no goals
                                      -/
                                  /-
                                    🎉 no goals
                                  -/
                      /-
                        ⊢ ∀ {α β : Type u_1} (x : Erased α) (y : Erased β), Eq (SeqRight.seqRight x fu …
                      -/
                   /-
                     ⊢ ∀ {α β : Type u_1} (f : Erased (α → β)) (x : Erased α), Eq (Bind.bind f fun  …
                   -/
                                   /-
                                     🎉 no goals
                                   -/
                   /-
                     ⊢ ∀ {α β : Type u_1} (g : α → β) (x : Erased α), Eq (Seq.seq (Pure.pure g) fun …
                   -/
                     /-
                       ⊢ ∀ {α β γ : Type u_1} (x : Erased α) (f : α → Erased β) (g : β → Erased γ), E …
                     -/
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
    bind_assoc := by intros; ext; simp
                                  /-
                                    🎉 no goals
                                  -/
    bind_pure_comp := by intros; ext; simp
    bind_map := by intros; ext; simp [Seq.seq]
    seqLeft_eq := by intros; ext; simp [Seq.seq, Functor.mapConst, SeqLeft.seqLeft]
    seqRight_eq := by intros; ext; simp [Seq.seq, Functor.mapConst, SeqRight.seqRight]
    pure_seq := by intros; ext; simp [Seq.seq, Functor.mapConst, SeqRight.seqRight] }


