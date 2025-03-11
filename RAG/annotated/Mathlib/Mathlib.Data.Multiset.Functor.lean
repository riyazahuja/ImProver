instance functor : Functor Multiset where map := @map


@[simp]
theorem fmap_def {α' β'} {s : Multiset α'} (f : α' → β') : f <$> s = s.map f :=
  rfl


instance : LawfulFunctor Multiset where
               /-
                 ⊢ ∀ {α : Type u_1} (x : Multiset α), Eq (Functor.map id x) x
               -/
  id_map := by simp
               /-
                 🎉 no goals
               -/
                 /-
                   ⊢ ∀ {α β γ : Type u_1} (g : α → β) (h : β → γ) (x : Multiset α), Eq (Functor.m …
                 -/
  comp_map := by simp
                 /-
                   🎉 no goals
                 -/
  map_const {_ _} := rfl


/-- Map each element of a `Multiset` to an action, evaluate these actions in order,
    and collect the results.
-/
def traverse : Multiset α' → F (Multiset β') := by
  /-
    F : Type u → Type u
    inst✝¹ : Applicative F
    inst✝ : CommApplicative F
    α' β' : Type u
    f : α' → F β'
    ⊢ Multiset α' → F (Multiset β')
  -/
  refine Quotient.lift (Functor.map ofList ∘ Traversable.traverse f) ?_
  /-
    F : Type u → Type u
    inst✝¹ : Applicative F
    inst✝ : CommApplicative F
    α' β' : Type u
    f : α' → F β'
    ⊢ ∀ (a b : List α'), HasEquiv.Equiv a b → Eq (Function.comp (Functor.map Multi …
  -/
  introv p; unfold Function.comp
  induction p with
  | nil => rfl
  | @cons x l₁ l₂ _ h =>
    have :
      Multiset.cons <$> f x <*> ofList <$> Traversable.traverse f l₁ =
        Multiset.cons <$> f x <*> ofList <$> Traversable.traverse f l₂ := by
      rw [h]
    simpa [functor_norm] using this
  | swap x y l =>
    have :
      (fun a b (l : List β') ↦ (↑(a :: b :: l) : Multiset β')) <$> f y <*> f x =
        (fun a b l ↦ ↑(a :: b :: l)) <$> f x <*> f y := by
      rw [CommApplicative.commutative_map]
      congr
      funext a b l
      simpa [flip] using Perm.swap a b l
    simp [Function.comp_def, this, functor_norm]
  | trans => simp [*]


instance : Monad Multiset :=
  { Multiset.functor with
    pure := fun x ↦ {x}
    bind := @bind }


@[simp]
theorem pure_def {α} : (pure : α → Multiset α) = singleton :=
  rfl


@[simp]
theorem bind_def {α β} : (· >>= ·) = @bind α β :=
  rfl


                                   /-
                                     F : Type u → Type u
                                     inst✝¹ : Applicative F
                                     inst✝ : CommApplicative F
                                     α' β' : Type u
                                     f : α' → F β'
                                     ⊢ ∀ {α β : Type u_1} (x : α) (y : Multiset β), Eq (Functor.mapConst x y) (Func …
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
                        /-
                          F : Type u → Type u
                          inst✝¹ : Applicative F
                          inst✝ : CommApplicative F
                          α' β' : Type u
                          f : α' → F β'
                          α✝ : Type u_1
                          x✝ : Multiset α✝
                          ⊢ Eq (Functor.map id x✝) x✝
                        -/
                                   /-
                                     🎉 no goals
                                   -/
                        /-
                          🎉 no goals
                        -/
                             /-
                               F : Type u → Type u
                               inst✝¹ : Applicative F
                               inst✝ : CommApplicative F
                               α' β' : Type u
                               f : α' → F β'
                               α✝ β✝ : Type u_1
                               x✝¹ : α✝
                               x✝ : α✝ → Multiset β✝
                               ⊢ Eq (Bind.bind (Pure.pure x✝¹) x✝) (x✝ x✝¹)
                             -/
                                   /-
                                     🎉 no goals
                                   -/
                             /-
                               🎉 no goals
                             -/
                                  /-
                                    F : Type u → Type u
                                    inst✝¹ : Applicative F
                                    inst✝ : CommApplicative F
                                    α' β' : Type u
                                    f : α' → F β'
                                    α✝ β✝ : Type u_1
                                    x✝¹ : α✝ → β✝
                                    x✝ : Multiset α✝
                                    ⊢ Eq (Bind.bind x✝ fun y => Pure.pure (x✝¹ y)) (Functor.map x✝¹ x✝)
                                  -/
instance : LawfulMonad Multiset := LawfulMonad.mk'
                                  /-
                                    🎉 no goals
                                  -/
                                   /-
                                     🎉 no goals
                                   -/
  (bind_pure_comp := fun _ _ ↦ by simp only [pure_def, bind_def, bind_singleton, fmap_def])
  (id_map := fun _ ↦ by simp only [fmap_def, id_eq, map_id'])
  (pure_bind := fun _ _ ↦ by simp only [pure_def, bind_def, singleton_bind])
  (bind_assoc := @bind_assoc)


@[simp]
theorem map_comp_coe {α β} (h : α → β) :
    Functor.map h ∘ ofList = (ofList ∘ Functor.map h : List α → Multiset β) := by
  /-
    α β : Type u_1
    h : α → β
    ⊢ Eq (Function.comp (Functor.map h) Multiset.ofList) (Function.comp Multiset.o …
  -/
  funext; simp only [Function.comp_apply, fmap_def, map_coe, List.map_eq_map]
          /-
            🎉 no goals
          -/


theorem id_traverse {α : Type*} (x : Multiset α) : traverse (pure : α → Id α) x = x := by
  /-
    α : Type u_1
    x : Multiset α
    ⊢ Eq (Multiset.traverse Pure.pure x) x
  -/
  refine Quotient.inductionOn x ?_
  /-
    α : Type u_1
    x : Multiset α
    ⊢ ∀ (a : List α), Eq (Multiset.traverse Pure.pure (Quotient.mk (List.isSetoid  …
  -/
  intro
  /-
    α : Type u_1
    x : Multiset α
    a✝ : List α
    ⊢ Eq (Multiset.traverse Pure.pure (Quotient.mk (List.isSetoid α) a✝)) (Quotien …
  -/
  simp [traverse]
  /-
    🎉 no goals
  -/


theorem comp_traverse {G H : Type _ → Type _} [Applicative G] [Applicative H] [CommApplicative G]
    [CommApplicative H] {α β γ : Type _} (g : α → G β) (h : β → H γ) (x : Multiset α) :
    traverse (Comp.mk ∘ Functor.map h ∘ g) x =
    Comp.mk (Functor.map (traverse h) (traverse g x)) := by
  /-
    G H : Type u_1 → Type u_1
    inst✝³ : Applicative G
    inst✝² : Applicative H
    inst✝¹ : CommApplicative G
    inst✝ : CommApplicative H
    α β γ : Type u_1
    g : α → G β
    h : β → H γ
    x : Multiset α
    ⊢ Eq (Multiset.traverse (Function.comp Functor.Comp.mk (Function.comp (Functor …
  -/
  refine Quotient.inductionOn x ?_
  /-
    G H : Type u_1 → Type u_1
    inst✝³ : Applicative G
    inst✝² : Applicative H
    inst✝¹ : CommApplicative G
    inst✝ : CommApplicative H
    α β γ : Type u_1
    g : α → G β
    h : β → H γ
    x : Multiset α
    ⊢ ∀ (a : List α), Eq (Multiset.traverse (Function.comp Functor.Comp.mk (Functi …
  -/
  intro
  /-
    G H : Type u_1 → Type u_1
    inst✝³ : Applicative G
    inst✝² : Applicative H
    inst✝¹ : CommApplicative G
    inst✝ : CommApplicative H
    α β γ : Type u_1
    g : α → G β
    h : β → H γ
    x : Multiset α
    a✝ : List α
    ⊢ Eq (Multiset.traverse (Function.comp Functor.Comp.mk (Function.comp (Functor …
  -/
  simp only [traverse, quot_mk_to_coe, lift_coe, Function.comp_apply, Functor.map_map, functor_norm]
  /-
    🎉 no goals
  -/


theorem map_traverse {G : Type* → Type _} [Applicative G] [CommApplicative G] {α β γ : Type _}
    (g : α → G β) (h : β → γ) (x : Multiset α) :
    Functor.map (Functor.map h) (traverse g x) = traverse (Functor.map h ∘ g) x := by
  /-
    G : Type u_1 → Type u_1
    inst✝¹ : Applicative G
    inst✝ : CommApplicative G
    α β γ : Type u_1
    g : α → G β
    h : β → γ
    x : Multiset α
    ⊢ Eq (Functor.map (Functor.map h) (Multiset.traverse g x)) (Multiset.traverse  …
  -/
  refine Quotient.inductionOn x ?_
  /-
    G : Type u_1 → Type u_1
    inst✝¹ : Applicative G
    inst✝ : CommApplicative G
    α β γ : Type u_1
    g : α → G β
    h : β → γ
    x : Multiset α
    ⊢ ∀ (a : List α), Eq (Functor.map (Functor.map h) (Multiset.traverse g (Quotie …
  -/
  intro
  /-
    G : Type u_1 → Type u_1
    inst✝¹ : Applicative G
    inst✝ : CommApplicative G
    α β γ : Type u_1
    g : α → G β
    h : β → γ
    x : Multiset α
    a✝ : List α
    ⊢ Eq (Functor.map (Functor.map h) (Multiset.traverse g (Quotient.mk (List.isSe …
  -/
  simp only [traverse, quot_mk_to_coe, lift_coe, Function.comp_apply, Functor.map_map, map_comp_coe]
  /-
    G : Type u_1 → Type u_1
    inst✝¹ : Applicative G
    inst✝ : CommApplicative G
    α β γ : Type u_1
    g : α → G β
    h : β → γ
    x : Multiset α
    a✝ : List α
    ⊢ Eq (Functor.map (fun a => Functor.map h ↑a) (Traversable.traverse g a✝)) (Fu …
  -/
  rw [Traversable.map_traverse']
  /-
    G : Type u_1 → Type u_1
    inst✝¹ : Applicative G
    inst✝ : CommApplicative G
    α β γ : Type u_1
    g : α → G β
    h : β → γ
    x : Multiset α
    a✝ : List α
    ⊢ Eq (Functor.map (fun a => Functor.map h ↑a) (Traversable.traverse g a✝)) (Fu …
  -/
  simp only [fmap_def, Function.comp_apply, Functor.map_map, List.map_eq_map, map_coe]
  /-
    🎉 no goals
  -/


theorem traverse_map {G : Type* → Type _} [Applicative G] [CommApplicative G] {α β γ : Type _}
    (g : α → β) (h : β → G γ) (x : Multiset α) : traverse h (map g x) = traverse (h ∘ g) x := by
  /-
    G : Type u_1 → Type u_1
    inst✝¹ : Applicative G
    inst✝ : CommApplicative G
    α β γ : Type u_1
    g : α → β
    h : β → G γ
    x : Multiset α
    ⊢ Eq (Multiset.traverse h (Multiset.map g x)) (Multiset.traverse (Function.com …
  -/
  refine Quotient.inductionOn x ?_
  /-
    G : Type u_1 → Type u_1
    inst✝¹ : Applicative G
    inst✝ : CommApplicative G
    α β γ : Type u_1
    g : α → β
    h : β → G γ
    x : Multiset α
    ⊢ ∀ (a : List α), Eq (Multiset.traverse h (Multiset.map g (Quotient.mk (List.i …
  -/
  intro
  /-
    G : Type u_1 → Type u_1
    inst✝¹ : Applicative G
    inst✝ : CommApplicative G
    α β γ : Type u_1
    g : α → β
    h : β → G γ
    x : Multiset α
    a✝ : List α
    ⊢ Eq (Multiset.traverse h (Multiset.map g (Quotient.mk (List.isSetoid α) a✝))) …
  -/
  simp only [traverse, quot_mk_to_coe, map_coe, lift_coe, Function.comp_apply]
  /-
    G : Type u_1 → Type u_1
    inst✝¹ : Applicative G
    inst✝ : CommApplicative G
    α β γ : Type u_1
    g : α → β
    h : β → G γ
    x : Multiset α
    a✝ : List α
    ⊢ Eq (Functor.map Multiset.ofList (Traversable.traverse h (List.map g a✝))) (F …
  -/
  rw [← Traversable.traverse_map h g, List.map_eq_map]
  /-
    🎉 no goals
  -/


theorem naturality {G H : Type _ → Type _} [Applicative G] [Applicative H] [CommApplicative G]
    [CommApplicative H] (eta : ApplicativeTransformation G H) {α β : Type _} (f : α → G β)
    (x : Multiset α) : eta (traverse f x) = traverse (@eta _ ∘ f) x := by
  /-
    G H : Type u_1 → Type u_1
    inst✝³ : Applicative G
    inst✝² : Applicative H
    inst✝¹ : CommApplicative G
    inst✝ : CommApplicative H
    eta : ApplicativeTransformation G H
    α β : Type u_1
    f : α → G β
    x : Multiset α
    ⊢ Eq ((fun {α} => eta.app α) (Multiset.traverse f x)) (Multiset.traverse (Func …
  -/
  refine Quotient.inductionOn x ?_
  /-
    G H : Type u_1 → Type u_1
    inst✝³ : Applicative G
    inst✝² : Applicative H
    inst✝¹ : CommApplicative G
    inst✝ : CommApplicative H
    eta : ApplicativeTransformation G H
    α β : Type u_1
    f : α → G β
    x : Multiset α
    ⊢ ∀ (a : List α), Eq ((fun {α} => eta.app α) (Multiset.traverse f (Quotient.mk …
  -/
  intro
  simp only [quot_mk_to_coe, traverse, lift_coe, Function.comp_apply,
    ApplicativeTransformation.preserves_map, LawfulTraversable.naturality]


