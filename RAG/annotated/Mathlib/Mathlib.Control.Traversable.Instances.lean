theorem Option.id_traverse {α} (x : Option α) : Option.traverse (pure : α → Id α) x = x := by
  /-
    α : Type u_1
    x : Option α
    ⊢ Eq (Option.traverse Pure.pure x) x
  -/
              /-
                🎉 no goals
              -/
  cases x <;> rfl
              /-
                🎉 no goals
              -/


theorem Option.comp_traverse {α β γ} (f : β → F γ) (g : α → G β) (x : Option α) :
    Option.traverse (Comp.mk ∘ (f <$> ·) ∘ g) x =
      Comp.mk (Option.traverse f <$> Option.traverse g x) := by
  /-
    F G : Type u → Type u
    inst✝² : Applicative F
    inst✝¹ : Applicative G
    inst✝ : LawfulApplicative G
    α : Type u_1
    β γ : Type u
    f : β → F γ
    g : α → G β
    x : Option α
    ⊢ Eq (Option.traverse (Function.comp Functor.Comp.mk (Function.comp (fun x =>  …
  -/
                                        /-
                                          🎉 no goals
                                        -/
  cases x <;> (simp! [functor_norm] <;> rfl)
               /-
                 🎉 no goals
               -/


theorem Option.traverse_eq_map_id {α β} (f : α → β) (x : Option α) :
                                                                                  /-
                                                                                    α β : Type u_1
                                                                                    f : α → β
                                                                                    x : Option α
                                                                                    ⊢ Eq (Option.traverse (Function.comp Pure.pure f) x) (Pure.pure (Functor.map f …
                                                                                  -/
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/
    Option.traverse ((pure : _ → Id _) ∘ f) x = (pure : _ → Id _) (f <$> x) := by cases x <;> rfl
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


theorem Option.naturality [LawfulApplicative F] {α β} (f : α → F β) (x : Option α) :
    η (Option.traverse f x) = Option.traverse (@η _ ∘ f) x := by
  -- Porting note: added `ApplicativeTransformation` theorems
  /-
    F G : Type u → Type u
    inst✝³ : Applicative F
    inst✝² : Applicative G
    inst✝¹ : LawfulApplicative G
    η : ApplicativeTransformation F G
    inst✝ : LawfulApplicative F
    α : Type u_1
    β : Type u
    f : α → F β
    x : Option α
    ⊢ Eq ((fun {α} => η.app α) (Option.traverse f x)) (Option.traverse (Function.c …
  -/
  cases' x with x <;> simp! [*, functor_norm, ApplicativeTransformation.preserves_map,
    ApplicativeTransformation.preserves_seq, ApplicativeTransformation.preserves_pure]


instance : LawfulTraversable Option :=
  { show LawfulMonad Option from inferInstance with
    id_traverse := Option.id_traverse
    comp_traverse := Option.comp_traverse
    traverse_eq_map_id := Option.traverse_eq_map_id
    naturality := fun η _ _ f x => Option.naturality η f x }


protected theorem id_traverse {α} (xs : List α) : List.traverse (pure : α → Id α) xs = xs := by
  /-
    α : Type u_1
    xs : List α
    ⊢ Eq (List.traverse Pure.pure xs) xs
  -/
                   /-
                     🎉 no goals
                   -/
  induction xs <;> simp! [*, List.traverse, functor_norm]; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/


protected theorem comp_traverse {α β γ} (f : β → F γ) (g : α → G β) (x : List α) :
    List.traverse (Comp.mk ∘ (f <$> ·) ∘ g) x =
    Comp.mk (List.traverse f <$> List.traverse g x) := by
  /-
    F G : Type u → Type u
    inst✝² : Applicative F
    inst✝¹ : Applicative G
    inst✝ : LawfulApplicative G
    α : Type u_1
    β γ : Type u
    f : β → F γ
    g : α → G β
    x : List α
    ⊢ Eq (List.traverse (Function.comp Functor.Comp.mk (Function.comp (fun x => Fu …
  -/
                                              /-
                                                🎉 no goals
                                              -/
  induction x <;> simp! [*, functor_norm] <;> rfl
                                              /-
                                                🎉 no goals
                                              -/


protected theorem traverse_eq_map_id {α β} (f : α → β) (x : List α) :
    List.traverse ((pure : _ → Id _) ∘ f) x = (pure : _ → Id _) (f <$> x) := by
  /-
    α β : Type u_1
    f : α → β
    x : List α
    ⊢ Eq (List.traverse (Function.comp Pure.pure f) x) (Pure.pure (Functor.map f x))
  -/
                  /-
                    🎉 no goals
                  -/
  induction x <;> simp! [*, functor_norm]; rfl
                                           /-
                                             🎉 no goals
                                           -/


protected theorem naturality {α β} (f : α → F β) (x : List α) :
    η (List.traverse f x) = List.traverse (@η _ ∘ f) x := by
  -- Porting note: added `ApplicativeTransformation` theorems
  /-
    F G : Type u → Type u
    inst✝³ : Applicative F
    inst✝² : Applicative G
    inst✝¹ : LawfulApplicative G
    inst✝ : LawfulApplicative F
    η : ApplicativeTransformation F G
    α : Type u_1
    β : Type u
    f : α → F β
    x : List α
    ⊢ Eq ((fun {α} => η.app α) (List.traverse f x)) (List.traverse (Function.comp  …
  -/
  induction x <;> simp! [*, functor_norm, ApplicativeTransformation.preserves_map,
    ApplicativeTransformation.preserves_seq, ApplicativeTransformation.preserves_pure]


instance : LawfulTraversable.{u} List :=
  { show LawfulMonad List from inferInstance with
    id_traverse := List.id_traverse
    comp_traverse := List.comp_traverse
    traverse_eq_map_id := List.traverse_eq_map_id
    naturality := List.naturality }


@[simp]
theorem traverse_nil : traverse f ([] : List α') = (pure [] : F (List β')) :=
  rfl


@[simp]
theorem traverse_cons (a : α') (l : List α') :
    traverse f (a :: l) = (· :: ·) <$> f a <*> traverse f l :=
  rfl


@[simp]
theorem traverse_append :
    ∀ as bs : List α', traverse f (as ++ bs) = (· ++ ·) <$> traverse f as <*> traverse f bs
                 /-
                   F : Type u → Type u
                   inst✝¹ : Applicative F
                   α' β' : Type u
                   f : α' → F β'
                   inst✝ : LawfulApplicative F
                   bs : List α'
                   ⊢ Eq (Traversable.traverse f (HAppend.hAppend List.nil bs)) (Seq.seq (Functor. …
                 -/
  | [], bs => by simp [functor_norm]
                 /-
                   🎉 no goals
                 -/
                      /-
                        F : Type u → Type u
                        inst✝¹ : Applicative F
                        α' β' : Type u
                        f : α' → F β'
                        inst✝ : LawfulApplicative F
                        a : α'
                        as bs : List α'
                        ⊢ Eq (Traversable.traverse f (HAppend.hAppend (List.cons a as) bs)) (Seq.seq ( …
                      -/
  | a :: as, bs => by simp [traverse_append as bs, functor_norm]; congr
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem mem_traverse {f : α' → Set β'} :
    ∀ (l : List α') (n : List β'), n ∈ traverse f l ↔ Forall₂ (fun b a => b ∈ f a) n l
                 /-
                   α' β' : Type u
                   f : α' → Set β'
                   ⊢ Iff (Membership.mem (Traversable.traverse f List.nil) List.nil) (List.Forall …
                 -/
  | [], [] => by simp
                 /-
                   🎉 no goals
                 -/
                      /-
                        α' β' : Type u
                        f : α' → Set β'
                        a : α'
                        as : List α'
                        ⊢ Iff (Membership.mem (Traversable.traverse f (List.cons a as)) List.nil) (Lis …
                      -/
  | a :: as, [] => by simp
                      /-
                        🎉 no goals
                      -/
                      /-
                        α' β' : Type u
                        f : α' → Set β'
                        b : β'
                        bs : List β'
                        ⊢ Iff (Membership.mem (Traversable.traverse f List.nil) (List.cons b bs)) (Lis …
                      -/
  | [], b :: bs => by simp
                      /-
                        🎉 no goals
                      -/
                           /-
                             α' β' : Type u
                             f : α' → Set β'
                             a : α'
                             as : List α'
                             b : β'
                             bs : List β'
                             ⊢ Iff (Membership.mem (Traversable.traverse f (List.cons a as)) (List.cons b b …
                           -/
  | a :: as, b :: bs => by simp [mem_traverse as bs]
                           /-
                             🎉 no goals
                           -/


protected theorem traverse_map {α β γ : Type u} (g : α → β) (f : β → G γ) (x : σ ⊕ α) :
    Sum.traverse f (g <$> x) = Sum.traverse (f ∘ g) x := by
  /-
    σ : Type u
    G : Type u → Type u
    inst✝ : Applicative G
    α β γ : Type u
    g : α → β
    f : β → G γ
    x : Sum σ α
    ⊢ Eq (Sum.traverse f (Functor.map g x)) (Sum.traverse (Function.comp f g) x)
  -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  cases x <;> simp [Sum.traverse, id_map, functor_norm] <;> rfl
                                                            /-
                                                              🎉 no goals
                                                            -/


protected theorem id_traverse {σ α} (x : σ ⊕ α) :
                                               /-
                                                 σ α : Type u_1
                                                 x : Sum σ α
                                                 ⊢ Eq (Sum.traverse Pure.pure x) x
                                               -/
                                                           /-
                                                             🎉 no goals
                                                           -/
    Sum.traverse (pure : α → Id α) x = x := by cases x <;> rfl
                                                           /-
                                                             🎉 no goals
                                                           -/


protected theorem comp_traverse {α β γ : Type u} (f : β → F γ) (g : α → G β) (x : σ ⊕ α) :
    Sum.traverse (Comp.mk ∘ (f <$> ·) ∘ g) x =
    Comp.mk.{u} (Sum.traverse f <$> Sum.traverse g x) := by
  /-
    σ : Type u
    F G : Type u → Type u
    inst✝² : Applicative F
    inst✝¹ : Applicative G
    inst✝ : LawfulApplicative G
    α β γ : Type u
    f : β → F γ
    g : α → G β
    x : Sum σ α
    ⊢ Eq (Sum.traverse (Function.comp Functor.Comp.mk (Function.comp (fun x => Fun …
  -/
                                                              /-
                                                                🎉 no goals
                                                              -/
  cases x <;> (simp! [Sum.traverse, map_id, functor_norm] <;> rfl)
               /-
                 🎉 no goals
               -/


protected theorem traverse_eq_map_id {α β} (f : α → β) (x : σ ⊕ α) :
    Sum.traverse ((pure : _ → Id _) ∘ f) x = (pure : _ → Id _) (f <$> x) := by
  /-
    σ α β : Type u
    f : α → β
    x : Sum σ α
    ⊢ Eq (Sum.traverse (Function.comp Pure.pure f) x) (Pure.pure (Functor.map f x))
  -/
                                              /-
                                                🎉 no goals
                                              -/
  induction x <;> simp! [*, functor_norm] <;> rfl
                                              /-
                                                🎉 no goals
                                              -/


protected theorem map_traverse {α β γ} (g : α → G β) (f : β → γ) (x : σ ⊕ α) :
    (f <$> ·) <$> Sum.traverse g x = Sum.traverse (f <$> g ·) x := by
  /-
    σ : Type u
    G : Type u → Type u
    inst✝¹ : Applicative G
    inst✝ : LawfulApplicative G
    α : Type u_1
    β γ : Type u
    g : α → G β
    f : β → γ
    x : Sum σ α
    ⊢ Eq (Functor.map (fun x => Functor.map f x) (Sum.traverse g x)) (Sum.traverse …
  -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  cases x <;> simp [Sum.traverse, id_map, functor_norm] <;> congr
                                                            /-
                                                              🎉 no goals
                                                            -/


protected theorem naturality {α β} (f : α → F β) (x : σ ⊕ α) :
    η (Sum.traverse f x) = Sum.traverse (@η _ ∘ f) x := by
  -- Porting note: added `ApplicativeTransformation` theorems
  /-
    σ : Type u
    F G : Type u → Type u
    inst✝³ : Applicative F
    inst✝² : Applicative G
    inst✝¹ : LawfulApplicative G
    inst✝ : LawfulApplicative F
    η : ApplicativeTransformation F G
    α : Type u_1
    β : Type u
    f : α → F β
    x : Sum σ α
    ⊢ Eq ((fun {α} => η.app α) (Sum.traverse f x)) (Sum.traverse (Function.comp (f …
  -/
  cases x <;> simp! [Sum.traverse, functor_norm, ApplicativeTransformation.preserves_map,
    ApplicativeTransformation.preserves_seq, ApplicativeTransformation.preserves_pure]


instance {σ : Type u} : LawfulTraversable.{u} (Sum σ) :=
  { show LawfulMonad (Sum σ) from inferInstance with
    id_traverse := Sum.id_traverse
    comp_traverse := Sum.comp_traverse
    traverse_eq_map_id := Sum.traverse_eq_map_id
    naturality := Sum.naturality }


