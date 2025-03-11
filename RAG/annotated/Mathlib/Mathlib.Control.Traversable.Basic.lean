/-- A transformation between applicative functors.  It is a natural
transformation such that `app` preserves the `Pure.pure` and
`Functor.map` (`<*>`) operations. See
`ApplicativeTransformation.preserves_map` for naturality. -/
structure ApplicativeTransformation : Type max (u + 1) v w where
  /-- The function on objects defined by an `ApplicativeTransformation`. -/
  app : ∀ α : Type u, F α → G α
  /-- An `ApplicativeTransformation` preserves `pure`. -/
  preserves_pure' : ∀ {α : Type u} (x : α), app _ (pure x) = pure x
  /-- An `ApplicativeTransformation` intertwines `seq`. -/
  preserves_seq' : ∀ {α β : Type u} (x : F (α → β)) (y : F α), app _ (x <*> y) = app _ x <*> app _ y


instance : CoeFun (ApplicativeTransformation F G) fun _ => ∀ {α}, F α → G α :=
  ⟨fun η ↦ η.app _⟩


theorem app_eq_coe (η : ApplicativeTransformation F G) : η.app = η :=
  rfl


@[simp]
theorem coe_mk (f : ∀ α : Type u, F α → G α) (pp ps) :
    (ApplicativeTransformation.mk f @pp @ps) = f :=
  rfl


protected theorem congr_fun (η η' : ApplicativeTransformation F G) (h : η = η') {α : Type u}
    (x : F α) : η x = η' x :=
  congrArg (fun η'' : ApplicativeTransformation F G => η'' x) h


protected theorem congr_arg (η : ApplicativeTransformation F G) {α : Type u} {x y : F α}
    (h : x = y) : η x = η y :=
  congrArg (fun z : F α => η z) h


theorem coe_inj ⦃η η' : ApplicativeTransformation F G⦄ (h : (η : ∀ α, F α → G α) = η') :
    η = η' := by
  /-
    F : Type u → Type v
    inst✝¹ : Applicative F
    G : Type u → Type w
    inst✝ : Applicative G
    η η' : ApplicativeTransformation F G
    h : Eq (fun {α} => η.app α) fun {α} => η'.app α
    ⊢ Eq η η'
  -/
  cases η
  /-
    case mk
    F : Type u → Type v
    inst✝¹ : Applicative F
    G : Type u → Type w
    inst✝ : Applicative G
    η' : ApplicativeTransformation F G
    app✝ : (α : Type u) → F α → G α
    preserves_pure'✝ : ∀ {α : Type u} (x : α), Eq (app✝ α (Pure.pure x)) (Pure.pur …
    preserves_seq'✝ : ∀ {α β : Type u} (x : F (α → β)) (y : F α), Eq (app✝ β (Seq. …
    h : Eq (fun {α} => { app := app✝, preserves_pure' := preserves_pure'✝, preserv …
    ⊢ Eq { app := app✝, preserves_pure' := preserves_pure'✝, preserves_seq' := pre …
  -/
  cases η'
  /-
    case mk.mk
    F : Type u → Type v
    inst✝¹ : Applicative F
    G : Type u → Type w
    inst✝ : Applicative G
    app✝¹ : (α : Type u) → F α → G α
    preserves_pure'✝¹ : ∀ {α : Type u} (x : α), Eq (app✝¹ α (Pure.pure x)) (Pure.p …
    preserves_seq'✝¹ : ∀ {α β : Type u} (x : F (α → β)) (y : F α), Eq (app✝¹ β (Se …
    app✝ : (α : Type u) → F α → G α
    preserves_pure'✝ : ∀ {α : Type u} (x : α), Eq (app✝ α (Pure.pure x)) (Pure.pur …
    preserves_seq'✝ : ∀ {α β : Type u} (x : F (α → β)) (y : F α), Eq (app✝ β (Seq. …
    h : Eq (fun {α} => { app := app✝¹, preserves_pure' := preserves_pure'✝¹, prese …
    ⊢ Eq { app := app✝¹, preserves_pure' := preserves_pure'✝¹, preserves_seq' := p …
  -/
  congr
  /-
    🎉 no goals
  -/


@[ext]
theorem ext ⦃η η' : ApplicativeTransformation F G⦄ (h : ∀ (α : Type u) (x : F α), η x = η' x) :
    η = η' := by
  /-
    F : Type u → Type v
    inst✝¹ : Applicative F
    G : Type u → Type w
    inst✝ : Applicative G
    η η' : ApplicativeTransformation F G
    h : ∀ (α : Type u) (x : F α), Eq ((fun {α} => η.app α) x) ((fun {α} => η'.app  …
    ⊢ Eq η η'
  -/
  apply coe_inj
  /-
    case h
    F : Type u → Type v
    inst✝¹ : Applicative F
    G : Type u → Type w
    inst✝ : Applicative G
    η η' : ApplicativeTransformation F G
    h : ∀ (α : Type u) (x : F α), Eq ((fun {α} => η.app α) x) ((fun {α} => η'.app  …
    ⊢ Eq (fun {α} => η.app α) fun {α} => η'.app α
  -/
  ext1 α
  /-
    case h.h
    F : Type u → Type v
    inst✝¹ : Applicative F
    G : Type u → Type w
    inst✝ : Applicative G
    η η' : ApplicativeTransformation F G
    h : ∀ (α : Type u) (x : F α), Eq ((fun {α} => η.app α) x) ((fun {α} => η'.app  …
    α : Type u
    ⊢ Eq (η.app α) (η'.app α)
  -/
  exact funext (h α)
  /-
    🎉 no goals
  -/


@[functor_norm]
theorem preserves_pure {α} : ∀ x : α, η (pure x) = pure x :=
  η.preserves_pure'


@[functor_norm]
theorem preserves_seq {α β : Type u} : ∀ (x : F (α → β)) (y : F α), η (x <*> y) = η x <*> η y :=
  η.preserves_seq'


@[functor_norm]
theorem preserves_map {α β} (x : α → β) (y : F α) : η (x <$> y) = x <$> η y := by
  /-
    F : Type u → Type v
    inst✝³ : Applicative F
    G : Type u → Type w
    inst✝² : Applicative G
    η : ApplicativeTransformation F G
    inst✝¹ : LawfulApplicative F
    inst✝ : LawfulApplicative G
    α β : Type u
    x : α → β
    y : F α
    ⊢ Eq ((fun {α} => η.app α) (Functor.map x y)) (Functor.map x ((fun {α} => η.ap …
  -/
  rw [← pure_seq, η.preserves_seq, preserves_pure, pure_seq]
  /-
    🎉 no goals
  -/


theorem preserves_map' {α β} (x : α → β) : @η _ ∘ Functor.map x = Functor.map x ∘ @η _ := by
  /-
    F : Type u → Type v
    inst✝³ : Applicative F
    G : Type u → Type w
    inst✝² : Applicative G
    η : ApplicativeTransformation F G
    inst✝¹ : LawfulApplicative F
    inst✝ : LawfulApplicative G
    α β : Type u
    x : α → β
    ⊢ Eq (Function.comp (fun {α} => η.app α) (Functor.map x)) (Function.comp (Func …
  -/
  ext y
  /-
    case h
    F : Type u → Type v
    inst✝³ : Applicative F
    G : Type u → Type w
    inst✝² : Applicative G
    η : ApplicativeTransformation F G
    inst✝¹ : LawfulApplicative F
    inst✝ : LawfulApplicative G
    α β : Type u
    x : α → β
    y : F α
    ⊢ Eq (Function.comp (fun {α} => η.app α) (Functor.map x) y) (Function.comp (Fu …
  -/
  exact preserves_map η x y
  /-
    🎉 no goals
  -/


/-- The identity applicative transformation from an applicative functor to itself. -/
def idTransformation : ApplicativeTransformation F F where
  app _ := id
                        /-
                          F : Type u → Type v
                          inst✝¹ : Applicative F
                          G : Type u → Type w
                          inst✝ : Applicative G
                          ⊢ ∀ {α : Type u} (x : α), Eq ((fun x => id) α (Pure.pure x)) (Pure.pure x)
                        -/
  preserves_pure' := by simp
                        /-
                          🎉 no goals
                        -/
                           /-
                             F : Type u → Type v
                             inst✝¹ : Applicative F
                             G : Type u → Type w
                             inst✝ : Applicative G
                             α✝ β✝ : Type u
                             x : F (α✝ → β✝)
                             y : F α✝
                             ⊢ Eq ((fun x => id) β✝ (Seq.seq x fun x => y)) (Seq.seq ((fun x => id) (α✝ → β …
                           -/
  preserves_seq' x y := by simp
                           /-
                             🎉 no goals
                           -/


instance : Inhabited (ApplicativeTransformation F F) :=
  ⟨idTransformation⟩


/-- The composition of applicative transformations. -/
def comp (η' : ApplicativeTransformation G H) (η : ApplicativeTransformation F G) :
    ApplicativeTransformation F H where
  app _ x := η' (η x)
  -- Porting note: something has gone wrong with `simp [functor_norm]`,
  -- which should suffice for the next two.
                          /-
                            F : Type u → Type v
                            inst✝² : Applicative F
                            G : Type u → Type w
                            inst✝¹ : Applicative G
                            H : Type u → Type s
                            inst✝ : Applicative H
                            η' : ApplicativeTransformation G H
                            η : ApplicativeTransformation F G
                            α✝ : Type u
                            x : α✝
                            ⊢ Eq ((fun x x_1 => (fun {α} => η'.app α) ((fun {α} => η.app α) x_1)) α✝ (Pure …
                          -/
  preserves_pure' x := by simp only [preserves_pure]
                          /-
                            🎉 no goals
                          -/
                           /-
                             F : Type u → Type v
                             inst✝² : Applicative F
                             G : Type u → Type w
                             inst✝¹ : Applicative G
                             H : Type u → Type s
                             inst✝ : Applicative H
                             η' : ApplicativeTransformation G H
                             η : ApplicativeTransformation F G
                             α✝ β✝ : Type u
                             x : F (α✝ → β✝)
                             y : F α✝
                             ⊢ Eq ((fun x x_1 => (fun {α} => η'.app α) ((fun {α} => η.app α) x_1)) β✝ (Seq. …
                           -/
  preserves_seq' x y := by simp only [preserves_seq]
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem comp_apply (η' : ApplicativeTransformation G H) (η : ApplicativeTransformation F G)
    {α : Type u} (x : F α) : η'.comp η x = η' (η x) :=
  rfl

-- Porting note: in mathlib3 we also had the assumption `[LawfulApplicative I]` because
-- this was assumed

theorem comp_assoc {I : Type u → Type t} [Applicative I]
    (η'' : ApplicativeTransformation H I) (η' : ApplicativeTransformation G H)
    (η : ApplicativeTransformation F G) : (η''.comp η').comp η = η''.comp (η'.comp η) :=
  rfl


@[simp]
theorem comp_id (η : ApplicativeTransformation F G) : η.comp idTransformation = η :=
  ext fun _ _ => rfl


@[simp]
theorem id_comp (η : ApplicativeTransformation F G) : idTransformation.comp η = η :=
  ext fun _ _ => rfl


/-- A traversable functor is a functor along with a way to commute
with all applicative functors (see `sequence`).  For example, if `t`
is the traversable functor `List` and `m` is the applicative functor
`IO`, then given a function `f : α → IO β`, the function `Functor.map f` is
`List α → List (IO β)`, but `traverse f` is `List α → IO (List β)`. -/
class Traversable (t : Type u → Type u) extends Functor t where
  /-- The function commuting a traversable functor `t` with an arbitrary applicative functor `m`. -/
  traverse : ∀ {m : Type u → Type u} [Applicative m] {α β}, (α → m β) → t α → m (t β)


/-- A traversable functor commutes with all applicative functors. -/
def sequence [Traversable t] : t (f α) → f (t α) :=
  traverse id


/-- A traversable functor is lawful if its `traverse` satisfies a
number of additional properties.  It must send `pure : α → Id α` to `pure`,
send the composition of applicative functors to the composition of the
`traverse` of each, send each function `f` to `fun x ↦ f <$> x`, and
satisfy a naturality condition with respect to applicative
transformations. -/
class LawfulTraversable (t : Type u → Type u) [Traversable t] extends LawfulFunctor t :
    Prop where
  /-- `traverse` plays well with `pure` of the identity monad -/
  id_traverse : ∀ {α} (x : t α), traverse (pure : α → Id α) x = x
  /-- `traverse` plays well with composition of applicative functors. -/
  comp_traverse :
    ∀ {F G} [Applicative F] [Applicative G] [LawfulApplicative F] [LawfulApplicative G] {α β γ}
      (f : β → F γ) (g : α → G β) (x : t α),
      traverse (Functor.Comp.mk ∘ map f ∘ g) x = Comp.mk (map (traverse f) (traverse g x))
  /-- An axiom for `traverse` involving `pure : β → Id β`. -/
  traverse_eq_map_id : ∀ {α β} (f : α → β) (x : t α),
    traverse ((pure : β → Id β) ∘ f) x = id.mk (f <$> x)
  /-- The naturality axiom explaining how lawful traversable functors should play with
  lawful applicative functors. -/
  naturality :
    ∀ {F G} [Applicative F] [Applicative G] [LawfulApplicative F] [LawfulApplicative G]
      (η : ApplicativeTransformation F G) {α β} (f : α → F β) (x : t α),
      η (traverse f x) = traverse (@η _ ∘ f) x


instance : Traversable Id :=
  ⟨id⟩


instance : LawfulTraversable Id where
  id_traverse _ := rfl
  comp_traverse _ _ _ := rfl
  traverse_eq_map_id _ _ := rfl
  naturality _ _ _ _ _ := rfl


instance : Traversable Option :=
  ⟨Option.traverse⟩


instance : Traversable List :=
  ⟨List.traverse⟩


/-- Defines a `traverse` function on the second component of a sum type.
This is used to give a `Traversable` instance for the functor `σ ⊕ -`. -/
protected def traverse {α β} (f : α → F β) : σ ⊕ α → F (σ ⊕ β)
  | Sum.inl x => pure (Sum.inl x)
  | Sum.inr x => Sum.inr <$> f x


instance {σ : Type u} : Traversable.{u} (Sum σ) :=
  ⟨@Sum.traverse _⟩

