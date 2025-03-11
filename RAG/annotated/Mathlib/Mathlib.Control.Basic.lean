/-- A generalization of `List.zipWith` which combines list elements with an `Applicative`. -/
def zipWithM {α₁ α₂ φ : Type u} (f : α₁ → α₂ → F φ) : ∀ (_ : List α₁) (_ : List α₂), F (List φ)
  | x :: xs, y :: ys => (· :: ·) <$> f x y <*> zipWithM f xs ys
  | _, _ => pure []


/-- Like `zipWithM` but evaluates the result as it traverses the lists using `*>`. -/
def zipWithM' (f : α → β → F γ) : List α → List β → F PUnit
  | x :: xs, y :: ys => f x y *> zipWithM' f xs ys
  | [], _ => pure PUnit.unit
  | _, [] => pure PUnit.unit


@[simp]
theorem pure_id'_seq (x : F α) : (pure fun x => x) <*> x = x :=
  pure_id_seq x


@[functor_norm]
theorem seq_map_assoc (x : F (α → β)) (f : γ → α) (y : F γ) :
    x <*> f <$> y = (· ∘ f) <$> x <*> y := by
  /-
    α β γ : Type u
    F : Type u → Type v
    inst✝¹ : Applicative F
    inst✝ : LawfulApplicative F
    x : F (α → β)
    f : γ → α
    y : F γ
    ⊢ Eq (Seq.seq x fun x => Functor.map f y) (Seq.seq (Functor.map (fun x => Func …
  -/
  simp only [← pure_seq]
  /-
    α β γ : Type u
    F : Type u → Type v
    inst✝¹ : Applicative F
    inst✝ : LawfulApplicative F
    x : F (α → β)
    f : γ → α
    y : F γ
    ⊢ Eq (Seq.seq x fun x => Seq.seq (Pure.pure f) fun x => y) (Seq.seq (Seq.seq ( …
  -/
  simp only [seq_assoc, Function.comp, seq_pure, ← comp_map]
  /-
    α β γ : Type u
    F : Type u → Type v
    inst✝¹ : Applicative F
    inst✝ : LawfulApplicative F
    x : F (α → β)
    f : γ → α
    y : F γ
    ⊢ Eq (Seq.seq (Functor.map (Function.comp (fun h => h f) Function.comp) x) fun …
  -/
  simp [pure_seq]
  /-
    α β γ : Type u
    F : Type u → Type v
    inst✝¹ : Applicative F
    inst✝ : LawfulApplicative F
    x : F (α → β)
    f : γ → α
    y : F γ
    ⊢ Eq (Seq.seq (Functor.map (Function.comp (fun h => h f) Function.comp) x) fun …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[functor_norm]
theorem map_seq (f : β → γ) (x : F (α → β)) (y : F α) :
    f <$> (x <*> y) = (f ∘ ·) <$> x <*> y := by
  /-
    α β γ : Type u
    F : Type u → Type v
    inst✝¹ : Applicative F
    inst✝ : LawfulApplicative F
    f : β → γ
    x : F (α → β)
    y : F α
    ⊢ Eq (Functor.map f (Seq.seq x fun x => y)) (Seq.seq (Functor.map (fun x => Fu …
  -/
  simp only [← pure_seq]; simp [seq_assoc]
                          /-
                            🎉 no goals
                          -/


theorem seq_bind_eq (x : m α) {g : β → m γ} {f : α → β} :
    f <$> x >>= g = x >>= g ∘ f :=
  show bind (f <$> x) g = bind x (g ∘ f) by
    /-
      α β γ : Type u
      m : Type u → Type v
      inst✝¹ : Monad m
      inst✝ : LawfulMonad m
      x : m α
      g : β → m γ
      f : α → β
      ⊢ Eq (Bind.bind (Functor.map f x) g) (Bind.bind x (Function.comp g f))
    -/
    rw [← bind_pure_comp, bind_assoc]
    /-
      α β γ : Type u
      m : Type u → Type v
      inst✝¹ : Monad m
      inst✝ : LawfulMonad m
      x : m α
      g : β → m γ
      f : α → β
      ⊢ Eq (Bind.bind x fun x => Bind.bind (Pure.pure (f x)) g) (Bind.bind x (Functi …
    -/
    simp [pure_bind, Function.comp_def]
    /-
      🎉 no goals
    -/
-- order of implicits and `Seq.seq` has a lazily evaluated second argument using `Unit`


@[functor_norm]
theorem fish_pure {α β} (f : α → m β) : f >=> pure = f := by
  /-
    m : Type u → Type v
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    α : Type u_1
    β : Type u
    f : α → m β
    ⊢ Eq (Bind.kleisliRight f Pure.pure) f
  -/
  simp (config := { unfoldPartialApp := true }) only [(· >=> ·), functor_norm]
  /-
    🎉 no goals
  -/


@[functor_norm]
theorem fish_pipe {α β} (f : α → m β) : pure >=> f = f := by
  /-
    m : Type u → Type v
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    α β : Type u
    f : α → m β
    ⊢ Eq (Bind.kleisliRight Pure.pure f) f
  -/
  simp (config := { unfoldPartialApp := true }) only [(· >=> ·), functor_norm]
  /-
    🎉 no goals
  -/

-- note: in Lean 3 `>=>` is left-associative, but in Lean 4 it is right-associative.

@[functor_norm]
theorem fish_assoc {α β γ φ} (f : α → m β) (g : β → m γ) (h : γ → m φ) :
    (f >=> g) >=> h = f >=> g >=> h := by
  /-
    m : Type u → Type v
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    α : Type u_1
    β γ φ : Type u
    f : α → m β
    g : β → m γ
    h : γ → m φ
    ⊢ Eq (Bind.kleisliRight (Bind.kleisliRight f g) h) (Bind.kleisliRight f (Bind. …
  -/
  simp (config := { unfoldPartialApp := true }) only [(· >=> ·), functor_norm]
  /-
    🎉 no goals
  -/


/-- Takes a value `β` and `List α` and accumulates pairs according to a monadic function `f`.
Accumulation occurs from the right (i.e., starting from the tail of the list). -/
def List.mapAccumRM (f : α → β' → m' (β' × γ')) : β' → List α → m' (β' × List γ')
  | a, [] => pure (a, [])
  | a, x :: xs => do
    let (a', ys) ← List.mapAccumRM f a xs
    let (a'', y) ← f x a'
    pure (a'', y :: ys)


/-- Takes a value `β` and `List α` and accumulates pairs according to a monadic function `f`.
Accumulation occurs from the left (i.e., starting from the head of the list). -/
def List.mapAccumLM (f : β' → α → m' (β' × γ')) : β' → List α → m' (β' × List γ')
  | a, [] => pure (a, [])
  | a, x :: xs => do
    let (a', y) ← f a x
    let (a'', ys) ← List.mapAccumLM f a' xs
    pure (a'', y :: ys)


theorem joinM_map_map {α β : Type u} (f : α → β) (a : m (m α)) :
    joinM (Functor.map f <$> a) = f <$> joinM a := by
  /-
    m : Type u → Type u
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    α β : Type u
    f : α → β
    a : m (m α)
    ⊢ Eq (joinM (Functor.map (Functor.map f) a)) (Functor.map f (joinM a))
  -/
  simp only [joinM, (· ∘ ·), id, ← bind_pure_comp, bind_assoc, map_bind, pure_bind]
  /-
    🎉 no goals
  -/


theorem joinM_map_joinM {α : Type u} (a : m (m (m α))) : joinM (joinM <$> a) = joinM (joinM a) := by
  /-
    m : Type u → Type u
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    α : Type u
    a : m (m (m α))
    ⊢ Eq (joinM (Functor.map joinM a)) (joinM (joinM a))
  -/
  simp only [joinM, (· ∘ ·), id, map_bind, ← bind_pure_comp, bind_assoc, pure_bind]
  /-
    🎉 no goals
  -/


@[simp]
theorem joinM_map_pure {α : Type u} (a : m α) : joinM (pure <$> a) = a := by
  /-
    m : Type u → Type u
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    α : Type u
    a : m α
    ⊢ Eq (joinM (Functor.map Pure.pure a)) a
  -/
  simp only [joinM, (· ∘ ·), id, map_bind, ← bind_pure_comp, bind_assoc, pure_bind, bind_pure]
  /-
    🎉 no goals
  -/


@[simp]
theorem joinM_pure {α : Type u} (a : m α) : joinM (pure a) = a :=
  LawfulMonad.pure_bind a id


/-- Returns `pure true` if the computation succeeds and `pure false` otherwise. -/
def succeeds {α} (x : F α) : F Bool :=
  Functor.mapConst true x <|> pure false


/-- Attempts to perform the computation, but fails silently if it doesn't succeed. -/
def tryM {α} (x : F α) : F Unit :=
  Functor.mapConst () x <|> pure ()


/-- Attempts to perform the computation, and returns `none` if it doesn't succeed. -/
def try? {α} (x : F α) : F (Option α) :=
  some <$> x <|> pure none


@[simp]
                                                                            /-
                                                                              F : Type → Type v
                                                                              inst✝ : Alternative F
                                                                              h : Decidable True
                                                                              ⊢ Eq (guard True) (Pure.pure Unit.unit)
                                                                            -/
theorem guard_true {h : Decidable True} : @guard F _ True h = pure () := by simp [guard, if_pos]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp]
theorem guard_false {h : Decidable False} : @guard F _ False h = failure := by
  /-
    F : Type → Type v
    inst✝ : Alternative F
    h : Decidable False
    ⊢ Eq (guard False) Alternative.failure
  -/
  simp [guard, if_neg not_false]
  /-
    🎉 no goals
  -/


/-- The monadic `bind` operation for `Sum`. -/
protected def bind {α β} : e ⊕ α → (α → e ⊕ β) → e ⊕ β
  | inl x, _ => inl x
  | inr x, f => f x
-- incorrectly marked as a bad translation by mathport, so we do not mark with `ₓ`.


instance : Monad (Sum.{v, u} e) where
  pure := @Sum.inr e
  bind := @Sum.bind e


instance : LawfulFunctor (Sum.{v, u} e) := by
  /-
    α β γ : Type u
    e : Type v
    ⊢ LawfulFunctor (Sum e)
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
  constructor <;> intros <;> (try casesm Sum _ _) <;> rfl
                                                      /-
                                                        🎉 no goals
                                                      -/


instance : LawfulMonad (Sum.{v, u} e) where
  seqRight_eq := by
    /-
      α β γ : Type u
      e : Type v
      ⊢ ∀ {α β : Type u} (x : Sum e α) (y : Sum e β), Eq (SeqRight.seqRight x fun x  …
    -/
    intros
    /-
      α β γ : Type u
      e : Type v
      α✝ β✝ : Type u
      x✝ : Sum e α✝
      y✝ : Sum e β✝
      ⊢ Eq (SeqRight.seqRight x✝ fun x => y✝) (Seq.seq (Functor.map (Function.const  …
    -/
    /-
      α β γ : Type u
      e : Type v
      ⊢ ∀ {α β : Type u} (x : Sum e α) (y : Sum e β), Eq (SeqLeft.seqLeft x fun x => …
    -/
                                          /-
                                            🎉 no goals
                                          -/
    /-
      α β γ : Type u
      e : Type v
      α✝ β✝ : Type u
      x✝ : Sum e α✝
      y✝ : Sum e β✝
      ⊢ Eq (SeqLeft.seqLeft x✝ fun x => y✝) (Seq.seq (Functor.map (Function.const β✝ …
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
    casesm Sum _ _ <;> casesm Sum _ _ <;> rfl
                                          /-
                                            🎉 no goals
                                          -/
  seqLeft_eq := by
    intros
    casesm Sum _ _ <;> rfl
  pure_seq := by
    /-
      α β γ : Type u
      e : Type v
      ⊢ ∀ {α β : Type u} (g : α → β) (x : Sum e α), Eq (Seq.seq (Pure.pure g) fun x_ …
    -/
    intros
    /-
      α β γ : Type u
      e : Type v
      α✝ β✝ : Type u
      g✝ : α✝ → β✝
      x✝ : Sum e α✝
      ⊢ Eq (Seq.seq (Pure.pure g✝) fun x => x✝) (Functor.map g✝ x✝)
    -/
    rfl
    /-
      🎉 no goals
    -/
  bind_assoc := by
    /-
      α β γ : Type u
      e : Type v
      ⊢ ∀ {α β γ : Type u} (x : Sum e α) (f : α → Sum e β) (g : β → Sum e γ), Eq (Bi …
    -/
    intros
    /-
      α β γ : Type u
      e : Type v
      α✝ β✝ γ✝ : Type u
      x✝ : Sum e α✝
      f✝ : α✝ → Sum e β✝
      g✝ : β✝ → Sum e γ✝
      ⊢ Eq (Bind.bind (Bind.bind x✝ f✝) g✝) (Bind.bind x✝ fun x => Bind.bind (f✝ x)  …
    -/
    /-
      α β γ : Type u
      e : Type v
      ⊢ ∀ {α β : Type u} (x : α) (f : α → Sum e β), Eq (Bind.bind (Pure.pure x) f) ( …
    -/
                       /-
                         🎉 no goals
                       -/
    /-
      α β γ : Type u
      e : Type v
      α✝ β✝ : Type u
      x✝ : α✝
      f✝ : α✝ → Sum e β✝
      ⊢ Eq (Bind.bind (Pure.pure x✝) f✝) (f✝ x✝)
    -/
    /-
      α β γ : Type u
      e : Type v
      ⊢ ∀ {α β : Type u} (f : α → β) (x : Sum e α), Eq (Bind.bind x fun a => Pure.pu …
    -/
    casesm Sum _ _ <;> rfl
    /-
      α β γ : Type u
      e : Type v
      α✝ β✝ : Type u
      f✝ : α✝ → β✝
      x✝ : Sum e α✝
      ⊢ Eq (Bind.bind x✝ fun a => Pure.pure (f✝ a)) (Functor.map f✝ x✝)
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
      α β γ : Type u
      e : Type v
      ⊢ ∀ {α β : Type u} (f : Sum e (α → β)) (x : Sum e α), Eq (Bind.bind f fun x_1  …
    -/
  pure_bind := by
    /-
      α β γ : Type u
      e : Type v
      α✝ β✝ : Type u
      f✝ : Sum e (α✝ → β✝)
      x✝ : Sum e α✝
      ⊢ Eq (Bind.bind f✝ fun x => Functor.map x x✝) (Seq.seq f✝ fun x => x✝)
    -/
                       /-
                         🎉 no goals
                       -/
    intros
                       /-
                         🎉 no goals
                       -/
    rfl
  bind_pure_comp := by
    intros
    casesm Sum _ _ <;> rfl
  bind_map := by
    intros
    casesm Sum _ _ <;> rfl


/-- A `CommApplicative` functor `m` is a (lawful) applicative functor which behaves identically on
`α × β` and `β × α`, so computations can occur in either order. -/
class CommApplicative (m : Type u → Type v) [Applicative m] extends LawfulApplicative m : Prop where
  /-- Computations performed first on `a : α` and then on `b : β` are equal to those performed in
  the reverse order. -/
  commutative_prod : ∀ {α β} (a : m α) (b : m β),
    Prod.mk <$> a <*> b = (fun (b : β) a => (a, b)) <$> b <*> a


theorem CommApplicative.commutative_map {m : Type u → Type v} [h : Applicative m]
    [CommApplicative m] {α β γ} (a : m α) (b : m β) {f : α → β → γ} :
  f <$> a <*> b = flip f <$> b <*> a :=
  calc
    f <$> a <*> b = (fun p : α × β => f p.1 p.2) <$> (Prod.mk <$> a <*> b) := by
      /-
        m : Type u → Type v
        h : Applicative m
        inst✝ : CommApplicative m
        α β γ : Type u
        a : m α
        b : m β
        f : α → β → γ
        ⊢ Eq (Seq.seq (Functor.map f a) fun x => b) (Functor.map (fun p => f p.fst p.s …
      -/
      simp only [map_seq, map_map, Function.comp_def]
      /-
        🎉 no goals
      -/
    _ = (fun b a => f a b) <$> b <*> a := by
      /-
        m : Type u → Type v
        h : Applicative m
        inst✝ : CommApplicative m
        α β γ : Type u
        a : m α
        b : m β
        f : α → β → γ
        ⊢ Eq (Functor.map (fun p => f p.fst p.snd) (Seq.seq (Functor.map Prod.mk a) fu …
      -/
      rw [@CommApplicative.commutative_prod m h]
      /-
        m : Type u → Type v
        h : Applicative m
        inst✝ : CommApplicative m
        α β γ : Type u
        a : m α
        b : m β
        f : α → β → γ
        ⊢ Eq (Functor.map (fun p => f p.fst p.snd) (Seq.seq (Functor.map (fun b a => { …
      -/
      simp [seq_map_assoc, map_seq, seq_assoc, seq_pure, map_map, (· ∘ ·)]
      /-
        m : Type u → Type v
        h : Applicative m
        inst✝ : CommApplicative m
        α β γ : Type u
        a : m α
        b : m β
        f : α → β → γ
        ⊢ Eq (Seq.seq (Functor.map (fun a => Function.comp (fun p => f p.fst p.snd) fu …
      -/
      rfl
      /-
        🎉 no goals
      -/

