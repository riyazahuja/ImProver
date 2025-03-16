structure MonadCont.Label (α : Type w) (m : Type u → Type v) (β : Type u) where
  apply : α → m β


def MonadCont.goto {α β} {m : Type u → Type v} (f : MonadCont.Label α m β) (x : α) :=
  f.apply x


class MonadCont (m : Type u → Type v) where
  callCC : ∀ {α β}, (MonadCont.Label α m β → m α) → m α


class LawfulMonadCont (m : Type u → Type v) [Monad m] [MonadCont m]
    extends LawfulMonad m : Prop where
  callCC_bind_right {α ω γ} (cmd : m α) (next : Label ω m γ → α → m ω) :
    (callCC fun f => cmd >>= next f) = cmd >>= fun x => callCC fun f => next f x
  callCC_bind_left {α} (β) (x : α) (dead : Label α m β → β → m α) :
    (callCC fun f : Label α m β => goto f x >>= dead f) = pure x
  callCC_dummy {α β} (dummy : m α) : (callCC fun _ : Label α m β => dummy) = dummy


def ContT (r : Type u) (m : Type u → Type v) (α : Type w) :=
  (α → m r) → m r


abbrev Cont (r : Type u) (α : Type w) :=
  ContT r id α


def run : ContT r m α → (α → m r) → m r :=
  id


def map (f : m r → m r) (x : ContT r m α) : ContT r m α :=
  f ∘ x


theorem run_contT_map_contT (f : m r → m r) (x : ContT r m α) : run (map f x) = f ∘ run x :=
  rfl


def withContT (f : (β → m r) → α → m r) (x : ContT r m α) : ContT r m β := fun g => x <| f g


theorem run_withContT (f : (β → m r) → α → m r) (x : ContT r m α) :
    run (withContT f x) = run x ∘ f :=
  rfl


@[ext]
protected theorem ext {x y : ContT r m α} (h : ∀ f, x.run f = y.run f) : x = y := by
  /-
    r : Type u
    m : Type u → Type v
    α : Type w
    x y : ContT r m α
    h : ∀ (f : α → m r), Eq (x.run f) (y.run f)
    ⊢ Eq x y
  -/
  unfold ContT; ext; apply h
                     /-
                       🎉 no goals
                     -/


instance : Monad (ContT r m) where
  pure x f := f x
  bind x f g := x fun i => f i g


                                      /-
                                        r : Type u
                                        m : Type u → Type v
                                        α β : Type w
                                        ⊢ ∀ {α β : Type u_1} (x : α) (y : ContT r m β), Eq (Functor.mapConst x y) (Fun …
                                      -/
                /-
                  r : Type u
                  m : Type u → Type v
                  α β : Type w
                  ⊢ ∀ {α : Type u_1} (x : ContT r m α), Eq (Functor.map id x) x
                -/
                                      /-
                                        🎉 no goals
                                      -/
                        /-
                          🎉 no goals
                        -/
                   /-
                     r : Type u
                     m : Type u → Type v
                     α β : Type w
                     ⊢ ∀ {α β : Type u_1} (x : α) (f : α → ContT r m β), Eq (Bind.bind (Pure.pure x …
                   -/
                                      /-
                                        🎉 no goals
                                      -/
                                /-
                                  🎉 no goals
                                -/
                    /-
                      r : Type u
                      m : Type u → Type v
                      α β : Type w
                      ⊢ ∀ {α β γ : Type u_1} (x : ContT r m α) (f : α → ContT r m β) (g : β → ContT  …
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
instance : LawfulMonad (ContT r m) := LawfulMonad.mk'
                                      /-
                                        🎉 no goals
                                      -/
  (id_map := by intros; rfl)
  (pure_bind := by intros; ext; rfl)
  (bind_assoc := by intros; ext; rfl)


def monadLift [Monad m] {α} : m α → ContT r m α := fun x f => x >>= f


instance [Monad m] : MonadLift m (ContT r m) where
  monadLift := ContT.monadLift


theorem monadLift_bind [Monad m] [LawfulMonad m] {α β} (x : m α) (f : α → m β) :
    (monadLift (x >>= f) : ContT r m β) = monadLift x >>= monadLift ∘ f := by
  /-
    r : Type u
    m : Type u → Type v
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    α β : Type u
    x : m α
    f : α → m β
    ⊢ Eq (ContT.monadLift (Bind.bind x f)) (Bind.bind (ContT.monadLift x) (Functio …
  -/
  ext
  simp only [monadLift, MonadLift.monadLift, (· ∘ ·), (· >>= ·), bind_assoc, id, run,
    ContT.monadLift]


instance : MonadCont (ContT r m) where
  callCC f g := f ⟨fun x _ => g x⟩ g


instance : LawfulMonadCont (ContT r m) where
                          /-
                            r : Type u
                            m : Type u → Type v
                            α β : Type w
                            ⊢ ∀ {α ω γ : Type u_1} (cmd : ContT r m α) (next : MonadCont.Label ω (ContT r  …
                          -/
  callCC_bind_right := by intros; ext; rfl
                                       /-
                                         🎉 no goals
                                       -/
                         /-
                           r : Type u
                           m : Type u → Type v
                           α β : Type w
                           ⊢ ∀ {α : Type u_1} (β : Type u_1) (x : α) (dead : MonadCont.Label α (ContT r m …
                         -/
  callCC_bind_left := by intros; ext; rfl
                                      /-
                                        🎉 no goals
                                      -/
                     /-
                       r : Type u
                       m : Type u → Type v
                       α β : Type w
                       ⊢ ∀ {α β : Type u_1} (dummy : ContT r m α), Eq (MonadCont.callCC fun x => dumm …
                     -/
  callCC_dummy := by intros; ext; rfl
                                  /-
                                    🎉 no goals
                                  -/


instance (ε) [MonadExcept ε m] : MonadExcept ε (ContT r m) where
  throw e _ := throw e
  tryCatch act h f := tryCatch (act f) fun e => h e f


def ExceptT.mkLabel {α β ε} : Label (Except.{u, u} ε α) m β → Label α (ExceptT ε m) β
  | ⟨f⟩ => ⟨fun a => monadLift <| f (Except.ok a)⟩


theorem ExceptT.goto_mkLabel {α β ε : Type _} (x : Label (Except.{u, u} ε α) m β) (i : α) :
    goto (ExceptT.mkLabel x) i = ExceptT.mk (Except.ok <$> goto x (Except.ok i)) := by
  /-
    m : Type u → Type v
    inst✝ : Monad m
    α β ε : Type u
    x : MonadCont.Label (Except ε α) m β
    i : α
    ⊢ Eq (MonadCont.goto (ExceptT.mkLabel x) i) (ExceptT.mk (Functor.map Except.ok …
  -/
  cases x; rfl
           /-
             🎉 no goals
           -/


nonrec def ExceptT.callCC {ε} [MonadCont m] {α β : Type _}
    (f : Label α (ExceptT ε m) β → ExceptT ε m α) : ExceptT ε m α :=
  ExceptT.mk (callCC fun x : Label _ m β => ExceptT.run <| f (ExceptT.mkLabel x))


instance {ε} [MonadCont m] : MonadCont (ExceptT ε m) where
  callCC := ExceptT.callCC


instance {ε} [MonadCont m] [LawfulMonadCont m] : LawfulMonadCont (ExceptT ε m) where
  callCC_bind_right := by
    /-
      m : Type u → Type v
      inst✝² : Monad m
      ε : Type u
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      ⊢ ∀ {α ω γ : Type u} (cmd : ExceptT ε m α) (next : MonadCont.Label ω (ExceptT  …
    -/
    intros; simp only [callCC, ExceptT.callCC, ExceptT.run_bind, callCC_bind_right]; ext
    /-
      case h
      m : Type u → Type v
      inst✝² : Monad m
      ε : Type u
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      α✝ ω✝ γ✝ : Type u
      cmd✝ : ExceptT ε m α✝
      next✝ : MonadCont.Label ω✝ (ExceptT ε m) γ✝ → α✝ → ExceptT ε m ω✝
      ⊢ Eq (ExceptT.mk (Bind.bind cmd✝.run fun x => MonadCont.callCC fun f => Except …
    -/
    dsimp
    /-
      case h
      m : Type u → Type v
      inst✝² : Monad m
      ε : Type u
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      α✝ ω✝ γ✝ : Type u
      cmd✝ : ExceptT ε m α✝
      next✝ : MonadCont.Label ω✝ (ExceptT ε m) γ✝ → α✝ → ExceptT ε m ω✝
      ⊢ Eq (Bind.bind cmd✝.run fun x => MonadCont.callCC fun f => ExceptT.run_bind.m …
    -/
                      /-
                        🎉 no goals
                      -/
    congr with ⟨⟩ <;> simp [ExceptT.bindCont, @callCC_dummy m _]
                      /-
                        🎉 no goals
                      -/
  callCC_bind_left := by
    /-
      m : Type u → Type v
      inst✝² : Monad m
      ε : Type u
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      ⊢ ∀ {α : Type u} (β : Type u) (x : α) (dead : MonadCont.Label α (ExceptT ε m)  …
    -/
    intros
    simp only [callCC, ExceptT.callCC, ExceptT.goto_mkLabel, map_eq_bind_pure_comp, Function.comp,
      ExceptT.run_bind, ExceptT.run_mk, bind_assoc, pure_bind, @callCC_bind_left m _]
    /-
      m : Type u → Type v
      inst✝² : Monad m
      ε : Type u
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      α✝ β✝ : Type u
      x✝ : α✝
      dead✝ : MonadCont.Label α✝ (ExceptT ε m) β✝ → β✝ → ExceptT ε m α✝
      ⊢ Eq (ExceptT.mk (Pure.pure (Except.ok x✝))) (Pure.pure x✝)
    -/
    ext; rfl
         /-
           🎉 no goals
         -/
                     /-
                       m : Type u → Type v
                       inst✝² : Monad m
                       ε : Type u
                       inst✝¹ : MonadCont m
                       inst✝ : LawfulMonadCont m
                       ⊢ ∀ {α β : Type u} (dummy : ExceptT ε m α), Eq (MonadCont.callCC fun x => dumm …
                     -/
  callCC_dummy := by intros; simp only [callCC, ExceptT.callCC, @callCC_dummy m _]; ext; rfl
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


def OptionT.mkLabel {α β} : Label (Option.{u} α) m β → Label α (OptionT m) β
  | ⟨f⟩ => ⟨fun a => monadLift <| f (some a)⟩


theorem OptionT.goto_mkLabel {α β : Type _} (x : Label (Option.{u} α) m β) (i : α) :
    goto (OptionT.mkLabel x) i = OptionT.mk (goto x (some i) >>= fun a => pure (some a)) :=
  rfl


nonrec def OptionT.callCC [MonadCont m] {α β : Type _} (f : Label α (OptionT m) β → OptionT m α) :
    OptionT m α :=
  OptionT.mk (callCC fun x : Label _ m β => OptionT.run <| f (OptionT.mkLabel x) : m (Option α))


instance [MonadCont m] : MonadCont (OptionT m) where
  callCC := OptionT.callCC


instance [MonadCont m] [LawfulMonadCont m] : LawfulMonadCont (OptionT m) where
  callCC_bind_right := by
    /-
      m : Type u → Type v
      inst✝² : Monad m
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      ⊢ ∀ {α ω γ : Type u} (cmd : OptionT m α) (next : MonadCont.Label ω (OptionT m) …
    -/
    intros; simp only [callCC, OptionT.callCC, OptionT.run_bind, callCC_bind_right]; ext
    /-
      case h
      m : Type u → Type v
      inst✝² : Monad m
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      α✝ ω✝ γ✝ : Type u
      cmd✝ : OptionT m α✝
      next✝ : MonadCont.Label ω✝ (OptionT m) γ✝ → α✝ → OptionT m ω✝
      ⊢ Eq (OptionT.mk (Bind.bind cmd✝.run fun x => MonadCont.callCC fun f => Option …
    -/
    dsimp
    /-
      case h
      m : Type u → Type v
      inst✝² : Monad m
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      α✝ ω✝ γ✝ : Type u
      cmd✝ : OptionT m α✝
      next✝ : MonadCont.Label ω✝ (OptionT m) γ✝ → α✝ → OptionT m ω✝
      ⊢ Eq (Bind.bind cmd✝.run fun x => MonadCont.callCC fun f => OptionT.run_bind.m …
    -/
                      /-
                        🎉 no goals
                      -/
    congr with ⟨⟩ <;> simp [@callCC_dummy m _]
                      /-
                        🎉 no goals
                      -/
  callCC_bind_left := by
    /-
      m : Type u → Type v
      inst✝² : Monad m
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      ⊢ ∀ {α : Type u} (β : Type u) (x : α) (dead : MonadCont.Label α (OptionT m) β  …
    -/
    intros
    simp only [callCC, OptionT.callCC, OptionT.goto_mkLabel, OptionT.run_bind, OptionT.run_mk,
      bind_assoc, pure_bind, @callCC_bind_left m _]
    /-
      m : Type u → Type v
      inst✝² : Monad m
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      α✝ β✝ : Type u
      x✝ : α✝
      dead✝ : MonadCont.Label α✝ (OptionT m) β✝ → β✝ → OptionT m α✝
      ⊢ Eq (OptionT.mk (Pure.pure (Option.some x✝))) (Pure.pure x✝)
    -/
    ext; rfl
         /-
           🎉 no goals
         -/
                     /-
                       m : Type u → Type v
                       inst✝² : Monad m
                       inst✝¹ : MonadCont m
                       inst✝ : LawfulMonadCont m
                       ⊢ ∀ {α β : Type u} (dummy : OptionT m α), Eq (MonadCont.callCC fun x => dummy) …
                     -/
  callCC_dummy := by intros; simp only [callCC, OptionT.callCC, @callCC_dummy m _]; ext; rfl
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/

/- Porting note: In Lean 3, `One ω` is required for `MonadLift (WriterT ω m)`. In Lean 4,
                 `EmptyCollection ω` or `Monoid ω` is required. So we give definitions for the both
                 instances. -/


def WriterT.mkLabel {α β ω} [EmptyCollection ω] : Label (α × ω) m β → Label α (WriterT ω m) β
  | ⟨f⟩ => ⟨fun a => monadLift <| f (a, ∅)⟩


def WriterT.mkLabel' {α β ω} [Monoid ω] : Label (α × ω) m β → Label α (WriterT ω m) β
  | ⟨f⟩ => ⟨fun a => monadLift <| f (a, 1)⟩


theorem WriterT.goto_mkLabel {α β ω : Type _} [EmptyCollection ω] (x : Label (α × ω) m β) (i : α) :
                                                                 /-
                                                                   m : Type u → Type v
                                                                   inst✝¹ : Monad m
                                                                   α : Type u_1
                                                                   β ω : Type u
                                                                   inst✝ : EmptyCollection ω
                                                                   x : MonadCont.Label (Prod α ω) m β
                                                                   i : α
                                                                   ⊢ Eq (MonadCont.goto (WriterT.mkLabel x) i) (MonadLiftT.monadLift (MonadCont.g …
                                                                 -/
    goto (WriterT.mkLabel x) i = monadLift (goto x (i, ∅)) := by cases x; rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem WriterT.goto_mkLabel' {α β ω : Type _} [Monoid ω] (x : Label (α × ω) m β) (i : α) :
                                                                  /-
                                                                    m : Type u → Type v
                                                                    inst✝¹ : Monad m
                                                                    α : Type u_1
                                                                    β ω : Type u
                                                                    inst✝ : Monoid ω
                                                                    x : MonadCont.Label (Prod α ω) m β
                                                                    i : α
                                                                    ⊢ Eq (MonadCont.goto (WriterT.mkLabel' x) i) (MonadLiftT.monadLift (MonadCont. …
                                                                  -/
    goto (WriterT.mkLabel' x) i = monadLift (goto x (i, 1)) := by cases x; rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


nonrec def WriterT.callCC [MonadCont m] {α β ω : Type _} [EmptyCollection ω]
    (f : Label α (WriterT ω m) β → WriterT ω m α) : WriterT ω m α :=
  WriterT.mk <| callCC (WriterT.run ∘ f ∘ WriterT.mkLabel : Label (α × ω) m β → m (α × ω))


def WriterT.callCC' [MonadCont m] {α β ω : Type _} [Monoid ω]
    (f : Label α (WriterT ω m) β → WriterT ω m α) : WriterT ω m α :=
  WriterT.mk <|
    MonadCont.callCC (WriterT.run ∘ f ∘ WriterT.mkLabel' : Label (α × ω) m β → m (α × ω))


instance (ω) [Monad m] [EmptyCollection ω] [MonadCont m] : MonadCont (WriterT ω m) where
  callCC := WriterT.callCC


instance (ω) [Monad m] [Monoid ω] [MonadCont m] : MonadCont (WriterT ω m) where
  callCC := WriterT.callCC'


def StateT.mkLabel {α β σ : Type u} : Label (α × σ) m (β × σ) → Label α (StateT σ m) β
  | ⟨f⟩ => ⟨fun a => StateT.mk (fun s => f (a, s))⟩


theorem StateT.goto_mkLabel {α β σ : Type u} (x : Label (α × σ) m (β × σ)) (i : α) :
                                                                         /-
                                                                           m : Type u → Type v
                                                                           α β σ : Type u
                                                                           x : MonadCont.Label (Prod α σ) m (Prod β σ)
                                                                           i : α
                                                                           ⊢ Eq (MonadCont.goto (StateT.mkLabel x) i) (StateT.mk fun s => MonadCont.goto  …
                                                                         -/
    goto (StateT.mkLabel x) i = StateT.mk (fun s => goto x (i, s)) := by cases x; rfl
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


nonrec def StateT.callCC {σ} [MonadCont m] {α β : Type _}
    (f : Label α (StateT σ m) β → StateT σ m α) : StateT σ m α :=
  StateT.mk (fun r => callCC fun f' => (f <| StateT.mkLabel f').run r)


instance {σ} [MonadCont m] : MonadCont (StateT σ m) where
  callCC := StateT.callCC


instance {σ} [Monad m] [MonadCont m] [LawfulMonadCont m] : LawfulMonadCont (StateT σ m) where
  callCC_bind_right := by
    /-
      m : Type u → Type v
      σ : Type u
      inst✝² : Monad m
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      ⊢ ∀ {α ω γ : Type u} (cmd : StateT σ m α) (next : MonadCont.Label ω (StateT σ  …
    -/
    intros
    /-
      m : Type u → Type v
      σ : Type u
      inst✝² : Monad m
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      α✝ ω✝ γ✝ : Type u
      cmd✝ : StateT σ m α✝
      next✝ : MonadCont.Label ω✝ (StateT σ m) γ✝ → α✝ → StateT σ m ω✝
      ⊢ Eq (MonadCont.callCC fun f => Bind.bind cmd✝ (next✝ f)) (Bind.bind cmd✝ fun  …
    -/
    simp only [callCC, StateT.callCC, StateT.run_bind, callCC_bind_right]; ext; rfl
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  callCC_bind_left := by
    /-
      m : Type u → Type v
      σ : Type u
      inst✝² : Monad m
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      ⊢ ∀ {α : Type u} (β : Type u) (x : α) (dead : MonadCont.Label α (StateT σ m) β …
    -/
    intros
    simp only [callCC, StateT.callCC, StateT.goto_mkLabel, StateT.run_bind, StateT.run_mk,
                         /-
                           m : Type u → Type v
                           σ : Type u
                           inst✝² : Monad m
                           inst✝¹ : MonadCont m
                           inst✝ : LawfulMonadCont m
                           α✝ β✝ : Type u
                           x✝ : α✝
                           dead✝ : MonadCont.Label α✝ (StateT σ m) β✝ → β✝ → StateT σ m α✝
                           ⊢ Eq (StateT.mk fun r => Pure.pure { fst := x✝, snd := r }) (Pure.pure x✝)
                         -/
      callCC_bind_left]; ext; rfl
                              /-
                                🎉 no goals
                              -/
  callCC_dummy := by
    /-
      m : Type u → Type v
      σ : Type u
      inst✝² : Monad m
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      ⊢ ∀ {α β : Type u} (dummy : StateT σ m α), Eq (MonadCont.callCC fun x => dummy …
    -/
    intros
    /-
      m : Type u → Type v
      σ : Type u
      inst✝² : Monad m
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      α✝ β✝ : Type u
      dummy✝ : StateT σ m α✝
      ⊢ Eq (MonadCont.callCC fun x => dummy✝) dummy✝
    -/
    simp only [callCC, StateT.callCC, @callCC_dummy m _]
    /-
      m : Type u → Type v
      σ : Type u
      inst✝² : Monad m
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      α✝ β✝ : Type u
      dummy✝ : StateT σ m α✝
      ⊢ Eq (StateT.mk fun r => dummy✝.run r) dummy✝
    -/
    ext; rfl
         /-
           🎉 no goals
         -/


def ReaderT.mkLabel {α β} (ρ) : Label α m β → Label α (ReaderT ρ m) β
  | ⟨f⟩ => ⟨monadLift ∘ f⟩


theorem ReaderT.goto_mkLabel {α ρ β} (x : Label α m β) (i : α) :
                                                              /-
                                                                m : Type u → Type v
                                                                α : Type u_1
                                                                ρ β : Type u
                                                                x : MonadCont.Label α m β
                                                                i : α
                                                                ⊢ Eq (MonadCont.goto (ReaderT.mkLabel ρ x) i) (MonadLiftT.monadLift (MonadCont …
                                                              -/
    goto (ReaderT.mkLabel ρ x) i = monadLift (goto x i) := by cases x; rfl
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


nonrec def ReaderT.callCC {ε} [MonadCont m] {α β : Type _}
    (f : Label α (ReaderT ε m) β → ReaderT ε m α) : ReaderT ε m α :=
  ReaderT.mk (fun r => callCC fun f' => (f <| ReaderT.mkLabel _ f').run r)


instance {ρ} [MonadCont m] : MonadCont (ReaderT ρ m) where
  callCC := ReaderT.callCC


instance {ρ} [Monad m] [MonadCont m] [LawfulMonadCont m] : LawfulMonadCont (ReaderT ρ m) where
                          /-
                            m : Type u → Type v
                            ρ : Type u
                            inst✝² : Monad m
                            inst✝¹ : MonadCont m
                            inst✝ : LawfulMonadCont m
                            ⊢ ∀ {α ω γ : Type u} (cmd : ReaderT ρ m α) (next : MonadCont.Label ω (ReaderT  …
                          -/
  callCC_bind_right := by intros; simp only [callCC, ReaderT.callCC, ReaderT.run_bind,
                                                        /-
                                                          m : Type u → Type v
                                                          ρ : Type u
                                                          inst✝² : Monad m
                                                          inst✝¹ : MonadCont m
                                                          inst✝ : LawfulMonadCont m
                                                          α✝ ω✝ γ✝ : Type u
                                                          cmd✝ : ReaderT ρ m α✝
                                                          next✝ : MonadCont.Label ω✝ (ReaderT ρ m) γ✝ → α✝ → ReaderT ρ m ω✝
                                                          ⊢ Eq (ReaderT.mk fun r => Bind.bind (cmd✝.run r) fun x => MonadCont.callCC fun …
                                                        -/
                                    callCC_bind_right]; ext; rfl
                                                             /-
                                                               🎉 no goals
                                                             -/
  callCC_bind_left := by
    /-
      m : Type u → Type v
      ρ : Type u
      inst✝² : Monad m
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      ⊢ ∀ {α : Type u} (β : Type u) (x : α) (dead : MonadCont.Label α (ReaderT ρ m)  …
    -/
    intros; simp only [callCC, ReaderT.callCC, ReaderT.goto_mkLabel, ReaderT.run_bind,
      ReaderT.run_monadLift, monadLift_self, callCC_bind_left]
    /-
      m : Type u → Type v
      ρ : Type u
      inst✝² : Monad m
      inst✝¹ : MonadCont m
      inst✝ : LawfulMonadCont m
      α✝ β✝ : Type u
      x✝ : α✝
      dead✝ : MonadCont.Label α✝ (ReaderT ρ m) β✝ → β✝ → ReaderT ρ m α✝
      ⊢ Eq (ReaderT.mk fun r => Pure.pure x✝) (Pure.pure x✝)
    -/
    ext; rfl
         /-
           🎉 no goals
         -/
                     /-
                       m : Type u → Type v
                       ρ : Type u
                       inst✝² : Monad m
                       inst✝¹ : MonadCont m
                       inst✝ : LawfulMonadCont m
                       ⊢ ∀ {α β : Type u} (dummy : ReaderT ρ m α), Eq (MonadCont.callCC fun x => dumm …
                     -/
  callCC_dummy := by intros; simp only [callCC, ReaderT.callCC, @callCC_dummy m _]; ext; rfl
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


/-- reduce the equivalence between two continuation passing monads to the equivalence between
their underlying monad -/
def ContT.equiv {m₁ : Type u₀ → Type v₀} {m₂ : Type u₁ → Type v₁} {α₁ r₁ : Type u₀}
    {α₂ r₂ : Type u₁} (F : m₁ r₁ ≃ m₂ r₂) (G : α₁ ≃ α₂) : ContT r₁ m₁ α₁ ≃ ContT r₂ m₂ α₂ where
  toFun f r := F <| f fun x => F.symm <| r <| G x
  invFun f r := F.symm <| f fun x => F <| r <| G.symm x
                   /-
                     m : Type u → Type v
                     m₁ : Type u₀ → Type v₀
                     m₂ : Type u₁ → Type v₁
                     α₁ r₁ : Type u₀
                     α₂ r₂ : Type u₁
                     F : Equiv (m₁ r₁) (m₂ r₂)
                     G : Equiv α₁ α₂
                     f : ContT r₁ m₁ α₁
                     ⊢ Eq ((fun f r => F.symm (f fun x => F (r (G.symm x)))) ((fun f r => F (f fun  …
                   -/
  left_inv f := by funext r; simp
                             /-
                               🎉 no goals
                             -/
                    /-
                      m : Type u → Type v
                      m₁ : Type u₀ → Type v₀
                      m₂ : Type u₁ → Type v₁
                      α₁ r₁ : Type u₀
                      α₂ r₂ : Type u₁
                      F : Equiv (m₁ r₁) (m₂ r₂)
                      G : Equiv α₁ α₂
                      f : ContT r₂ m₂ α₂
                      ⊢ Eq ((fun f r => F (f fun x => F.symm (r (G x)))) ((fun f r => F.symm (f fun  …
                    -/
  right_inv f := by funext r; simp
                              /-
                                🎉 no goals
                              -/

