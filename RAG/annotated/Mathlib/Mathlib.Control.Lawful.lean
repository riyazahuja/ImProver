protected def mk (f : σ → m (α × σ)) : StateT σ m α := f


@[simp]
theorem run_mk (f : σ → m (α × σ)) (st : σ) : StateT.run (StateT.mk f) st = f st :=
  rfl

-- Porting note: `StateT.adapt` is removed.


@[simp]
theorem run_monadLift {n} [Monad m] [MonadLiftT n m] (x : n α) :
    (monadLift x : ExceptT ε m α).run = Except.ok <$> (monadLift x : m α) :=
  rfl


@[simp]
theorem run_monadMap {n} [MonadFunctorT n m] (f : ∀ {α}, n α → n α) :
    (monadMap (@f) x : ExceptT ε m α).run = monadMap (@f) x.run :=
  rfl


protected def mk (f : σ → m α) : ReaderT σ m α := f


@[simp]
theorem run_mk (f : σ → m α) (r : σ) : ReaderT.run (ReaderT.mk f) r = f r :=
  rfl


@[ext] theorem ext {x x' : OptionT m α} (h : x.run = x'.run) : x = x' :=
  h

-- Porting note: This is proven by proj reduction in Lean 3.

@[simp]
theorem run_mk (x : m (Option α)) : OptionT.run (OptionT.mk x) = x :=
  rfl


@[simp]
theorem run_pure (a) : (pure a : OptionT m α).run = pure (some a) :=
  rfl


@[simp]
theorem run_bind (f : α → OptionT m β) :
    (x >>= f).run = x.run >>= fun
                              | some a => OptionT.run (f a)
                              | none   => pure none :=
  rfl


@[simp]
theorem run_map (f : α → β) [LawfulMonad m] : (f <$> x).run = Option.map f <$> x.run := by
  /-
    α β : Type u
    m : Type u → Type v
    x : OptionT m α
    inst✝¹ : Monad m
    f : α → β
    inst✝ : LawfulMonad m
    ⊢ Eq (Functor.map f x).run (Functor.map (Option.map f) x.run)
  -/
  rw [← bind_pure_comp _ x.run]
  change x.run >>= (fun
                     | some a => OptionT.run (pure (f a))
                     | none   => pure none) = _
  /-
    α β : Type u
    m : Type u → Type v
    x : OptionT m α
    inst✝¹ : Monad m
    f : α → β
    inst✝ : LawfulMonad m
    ⊢ Eq (Bind.bind x.run fun x => OptionT.run_bind.match_1 (fun x => m (Option β) …
  -/
  apply bind_congr
  /-
    case h
    α β : Type u
    m : Type u → Type v
    x : OptionT m α
    inst✝¹ : Monad m
    f : α → β
    inst✝ : LawfulMonad m
    ⊢ ∀ (a : Option α), Eq (OptionT.run_bind.match_1 (fun x => m (Option β)) a (fu …
  -/
                       /-
                         🎉 no goals
                       -/
  intro a; cases a <;> simp [Option.map, Option.bind]
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem run_monadLift {n} [MonadLiftT n m] (x : n α) :
    (monadLift x : OptionT m α).run = (monadLift x : m α) >>= fun a => pure (some a) :=
  rfl


@[simp]
theorem run_monadMap {n} [MonadFunctorT n m] (f : ∀ {α}, n α → n α) :
    (monadMap (@f) x : OptionT m α).run = monadMap (@f) x.run :=
  rfl


instance (m : Type u → Type v) [Monad m] [LawfulMonad m] : LawfulMonad (OptionT m) :=
  /-
    m : Type u → Type v
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    ⊢ ∀ {α β : Type u} (x : α) (y : OptionT m β), Eq (Functor.mapConst x y) (Funct …
  -/
  /-
    🎉 no goals
  -/
      /-
        m : Type u → Type v
        inst✝¹ : Monad m
        inst✝ : LawfulMonad m
        ⊢ ∀ {α : Type u} (x : OptionT m α), Eq (Functor.map id x) x
      -/
  /-
    🎉 no goals
  -/
      /-
        case h
        m : Type u → Type v
        inst✝¹ : Monad m
        inst✝ : LawfulMonad m
        α✝ : Type u
        x✝ : OptionT m α✝
        ⊢ Eq (Functor.map (Option.map id) x✝.run) x✝.run
      -/
  /-
    🎉 no goals
  -/
      /-
        case h
        m : Type u → Type v
        inst✝¹ : Monad m
        inst✝ : LawfulMonad m
        α✝ : Type u
        x✝ : OptionT m α✝
        ⊢ ∀ (a : Option α✝), Eq (Option.map id a) (id a)
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
  LawfulMonad.mk'
      /-
        m : Type u → Type v
        inst✝¹ : Monad m
        inst✝ : LawfulMonad m
        ⊢ ∀ {α β γ : Type u} (x : OptionT m α) (f : α → OptionT m β) (g : β → OptionT  …
      -/
  /-
    🎉 no goals
  -/
      /-
        case h
        m : Type u → Type v
        inst✝¹ : Monad m
        inst✝ : LawfulMonad m
        α✝ β✝ γ✝ : Type u
        x✝ : OptionT m α✝
        f✝ : α✝ → OptionT m β✝
        g✝ : β✝ → OptionT m γ✝
        ⊢ Eq (Bind.bind x✝.run fun x => Bind.bind (OptionT.run_bind.match_1 (fun x =>  …
      -/
                     /-
                       m : Type u → Type v
                       inst✝¹ : Monad m
                       inst✝ : LawfulMonad m
                       ⊢ ∀ {α β : Type u} (x : α) (f : α → OptionT m β), Eq (Bind.bind (Pure.pure x)  …
                     -/
    (id_map := by
                                                /-
                                                  🎉 no goals
                                                -/
      /-
        case h
        m : Type u → Type v
        inst✝¹ : Monad m
        inst✝ : LawfulMonad m
        α✝ β✝ γ✝ : Type u
        x✝ : OptionT m α✝
        f✝ : α✝ → OptionT m β✝
        g✝ : β✝ → OptionT m γ✝
        ⊢ ∀ (a : Option α✝), Eq (Bind.bind (OptionT.run_bind.match_1 (fun x => m (Opti …
      -/
                           /-
                             🎉 no goals
                           -/
      intros; apply OptionT.ext; simp only [OptionT.run_map]
                           /-
                             🎉 no goals
                           -/
      rw [map_congr, id_map]
      intro a; cases a <;> rfl)
    (bind_assoc := by
      intros; apply OptionT.ext; simp only [OptionT.run_bind, bind_assoc]
      rw [bind_congr]
      intro a; cases a <;> simp)
    (pure_bind := by intros; apply OptionT.ext; simp)

