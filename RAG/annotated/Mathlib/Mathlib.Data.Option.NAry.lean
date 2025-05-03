/-- The image of a binary function `f : α → β → γ` as a function `Option α → Option β → Option γ`.
Mathematically this should be thought of as the image of the corresponding function `α × β → γ`. -/
def map₂ (f : α → β → γ) (a : Option α) (b : Option β) : Option γ :=
  a.bind fun a => b.map <| f a


/-- `Option.map₂` in terms of monadic operations. Note that this can't be taken as the definition
because of the lack of universe polymorphism. -/
theorem map₂_def {α β γ : Type u} (f : α → β → γ) (a : Option α) (b : Option β) :
    map₂ f a b = f <$> a <*> b := by
  /-
    α β γ : Type u
    f : α → β → γ
    a : Option α
    b : Option β
    ⊢ Eq (Option.map₂ f a b) (Seq.seq (Functor.map f a) fun x => b)
  -/
              /-
                🎉 no goals
              -/
  cases a <;> rfl
              /-
                🎉 no goals
              -/


@[simp]
theorem map₂_some_some (f : α → β → γ) (a : α) (b : β) : map₂ f (some a) (some b) = f a b := rfl


theorem map₂_coe_coe (f : α → β → γ) (a : α) (b : β) : map₂ f a b = f a b := rfl


@[simp]
theorem map₂_none_left (f : α → β → γ) (b : Option β) : map₂ f none b = none := rfl


@[simp]
                                                                                    /-
                                                                                      α : Type u_1
                                                                                      β : Type u_2
                                                                                      γ : Type u_3
                                                                                      f : α → β → γ
                                                                                      a : Option α
                                                                                      ⊢ Eq (Option.map₂ f a Option.none) Option.none
                                                                                    -/
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/
theorem map₂_none_right (f : α → β → γ) (a : Option α) : map₂ f a none = none := by cases a <;> rfl
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


@[simp]
theorem map₂_coe_left (f : α → β → γ) (a : α) (b : Option β) : map₂ f a b = b.map fun b => f a b :=
  rfl

-- Porting note: This proof was `rfl` in Lean3, but now is not.

@[simp]
theorem map₂_coe_right (f : α → β → γ) (a : Option α) (b : β) :
                                            /-
                                              α : Type u_1
                                              β : Type u_2
                                              γ : Type u_3
                                              f : α → β → γ
                                              a : Option α
                                              b : β
                                              ⊢ Eq (Option.map₂ f a (Option.some b)) (Option.map (fun a => f a b) a)
                                            -/
                                                        /-
                                                          🎉 no goals
                                                        -/
    map₂ f a b = a.map fun a => f a b := by cases a <;> rfl
                                                        /-
                                                          🎉 no goals
                                                        -/

-- Porting note: Removed the `@[simp]` tag as membership of an `Option` is no-longer simp-normal.

theorem mem_map₂_iff {c : γ} : c ∈ map₂ f a b ↔ ∃ a' b', a' ∈ a ∧ b' ∈ b ∧ f a' b' = c := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β → γ
    a : Option α
    b : Option β
    c : γ
    ⊢ Iff (Membership.mem (Option.map₂ f a b) c) (Exists fun a' => Exists fun b' = …
  -/
  simp [map₂, bind_eq_some]
  /-
    🎉 no goals
  -/


@[simp]
theorem map₂_eq_none_iff : map₂ f a b = none ↔ a = none ∨ b = none := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β → γ
    a : Option α
    b : Option β
    ⊢ Iff (Eq (Option.map₂ f a b) Option.none) (Or (Eq a Option.none) (Eq b Option …
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
  cases a <;> cases b <;> simp
                          /-
                            🎉 no goals
                          -/


theorem map₂_swap (f : α → β → γ) (a : Option α) (b : Option β) :
                                                   /-
                                                     α : Type u_1
                                                     β : Type u_2
                                                     γ : Type u_3
                                                     f : α → β → γ
                                                     a : Option α
                                                     b : Option β
                                                     ⊢ Eq (Option.map₂ f a b) (Option.map₂ (fun a b => f b a) b a)
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
    map₂ f a b = map₂ (fun a b => f b a) b a := by cases a <;> cases b <;> rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem map_map₂ (f : α → β → γ) (g : γ → δ) :
                                                               /-
                                                                 α : Type u_1
                                                                 β : Type u_2
                                                                 γ : Type u_3
                                                                 δ : Type u_4
                                                                 a : Option α
                                                                 b : Option β
                                                                 f : α → β → γ
                                                                 g : γ → δ
                                                                 ⊢ Eq (Option.map g (Option.map₂ f a b)) (Option.map₂ (fun a b => g (f a b)) a b)
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
    (map₂ f a b).map g = map₂ (fun a b => g (f a b)) a b := by cases a <;> cases b <;> rfl
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


theorem map₂_map_left (f : γ → β → δ) (g : α → γ) :
                                                               /-
                                                                 α : Type u_1
                                                                 β : Type u_2
                                                                 γ : Type u_3
                                                                 δ : Type u_4
                                                                 a : Option α
                                                                 b : Option β
                                                                 f : γ → β → δ
                                                                 g : α → γ
                                                                 ⊢ Eq (Option.map₂ f (Option.map g a) b) (Option.map₂ (fun a b => f (g a) b) a b)
                                                               -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
    map₂ f (a.map g) b = map₂ (fun a b => f (g a) b) a b := by cases a <;> rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem map₂_map_right (f : α → γ → δ) (g : β → γ) :
                                                               /-
                                                                 α : Type u_1
                                                                 β : Type u_2
                                                                 γ : Type u_3
                                                                 δ : Type u_4
                                                                 a : Option α
                                                                 b : Option β
                                                                 f : α → γ → δ
                                                                 g : β → γ
                                                                 ⊢ Eq (Option.map₂ f a (Option.map g b)) (Option.map₂ (fun a b => f a (g b)) a b)
                                                               -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
    map₂ f a (b.map g) = map₂ (fun a b => f a (g b)) a b := by cases b <;> rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
theorem map₂_curry (f : α × β → γ) (a : Option α) (b : Option β) :
    map₂ (curry f) a b = Option.map f (map₂ Prod.mk a b) := (map_map₂ _ _).symm


@[simp]
theorem map_uncurry (f : α → β → γ) (x : Option (α × β)) :
                                                                       /-
                                                                         α : Type u_1
                                                                         β : Type u_2
                                                                         γ : Type u_3
                                                                         f : α → β → γ
                                                                         x : Option (Prod α β)
                                                                         ⊢ Eq (Option.map (Function.uncurry f) x) (Option.map₂ f (Option.map Prod.fst x …
                                                                       -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
    x.map (uncurry f) = map₂ f (x.map Prod.fst) (x.map Prod.snd) := by cases x <;> rfl
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


theorem map₂_assoc {f : δ → γ → ε} {g : α → β → δ} {f' : α → ε' → ε} {g' : β → γ → ε'}
    (h_assoc : ∀ a b c, f (g a b) c = f' a (g' b c)) :
    map₂ f (map₂ g a b) c = map₂ f' a (map₂ g' b c) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    a : Option α
    b : Option β
    c : Option γ
    ε : Type u_8
    ε' : Type u_9
    f : δ → γ → ε
    g : α → β → δ
    f' : α → ε' → ε
    g' : β → γ → ε'
    h_assoc : ∀ (a : α) (b : β) (c : γ), Eq (f (g a b) c) (f' a (g' b c))
    ⊢ Eq (Option.map₂ f (Option.map₂ g a b) c) (Option.map₂ f' a (Option.map₂ g' b …
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
  cases a <;> cases b <;> cases c <;> simp [h_assoc]
                                      /-
                                        🎉 no goals
                                      -/


theorem map₂_comm {g : β → α → γ} (h_comm : ∀ a b, f a b = g b a) : map₂ f a b = map₂ g b a := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β → γ
    a : Option α
    b : Option β
    g : β → α → γ
    h_comm : ∀ (a : α) (b : β), Eq (f a b) (g b a)
    ⊢ Eq (Option.map₂ f a b) (Option.map₂ g b a)
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
  cases a <;> cases b <;> simp [h_comm]
                          /-
                            🎉 no goals
                          -/


theorem map₂_left_comm {f : α → δ → ε} {g : β → γ → δ} {f' : α → γ → δ'} {g' : β → δ' → ε}
    (h_left_comm : ∀ a b c, f a (g b c) = g' b (f' a c)) :
    map₂ f a (map₂ g b c) = map₂ g' b (map₂ f' a c) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    a : Option α
    b : Option β
    c : Option γ
    δ' : Type u_7
    ε : Type u_8
    f : α → δ → ε
    g : β → γ → δ
    f' : α → γ → δ'
    g' : β → δ' → ε
    h_left_comm : ∀ (a : α) (b : β) (c : γ), Eq (f a (g b c)) (g' b (f' a c))
    ⊢ Eq (Option.map₂ f a (Option.map₂ g b c)) (Option.map₂ g' b (Option.map₂ f' a …
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
  cases a <;> cases b <;> cases c <;> simp [h_left_comm]
                                      /-
                                        🎉 no goals
                                      -/


theorem map₂_right_comm {f : δ → γ → ε} {g : α → β → δ} {f' : α → γ → δ'} {g' : δ' → β → ε}
    (h_right_comm : ∀ a b c, f (g a b) c = g' (f' a c) b) :
    map₂ f (map₂ g a b) c = map₂ g' (map₂ f' a c) b := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    a : Option α
    b : Option β
    c : Option γ
    δ' : Type u_7
    ε : Type u_8
    f : δ → γ → ε
    g : α → β → δ
    f' : α → γ → δ'
    g' : δ' → β → ε
    h_right_comm : ∀ (a : α) (b : β) (c : γ), Eq (f (g a b) c) (g' (f' a c) b)
    ⊢ Eq (Option.map₂ f (Option.map₂ g a b) c) (Option.map₂ g' (Option.map₂ f' a c …
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
  cases a <;> cases b <;> cases c <;> simp [h_right_comm]
                                      /-
                                        🎉 no goals
                                      -/


theorem map_map₂_distrib {g : γ → δ} {f' : α' → β' → δ} {g₁ : α → α'} {g₂ : β → β'}
    (h_distrib : ∀ a b, g (f a b) = f' (g₁ a) (g₂ b)) :
    (map₂ f a b).map g = map₂ f' (a.map g₁) (b.map g₂) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    f : α → β → γ
    a : Option α
    b : Option β
    α' : Type u_5
    β' : Type u_6
    g : γ → δ
    f' : α' → β' → δ
    g₁ : α → α'
    g₂ : β → β'
    h_distrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g₁ a) (g₂ b))
    ⊢ Eq (Option.map g (Option.map₂ f a b)) (Option.map₂ f' (Option.map g₁ a) (Opt …
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
  cases a <;> cases b <;> simp [h_distrib]
                          /-
                            🎉 no goals
                          -/


/-- Symmetric statement to `Option.map₂_map_left_comm`. -/
theorem map_map₂_distrib_left {g : γ → δ} {f' : α' → β → δ} {g' : α → α'}
    (h_distrib : ∀ a b, g (f a b) = f' (g' a) b) :
                                                    /-
                                                      α : Type u_1
                                                      β : Type u_2
                                                      γ : Type u_3
                                                      δ : Type u_4
                                                      f : α → β → γ
                                                      a : Option α
                                                      b : Option β
                                                      α' : Type u_5
                                                      g : γ → δ
                                                      f' : α' → β → δ
                                                      g' : α → α'
                                                      h_distrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g' a) b)
                                                      ⊢ Eq (Option.map g (Option.map₂ f a b)) (Option.map₂ f' (Option.map g' a) b)
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
    (map₂ f a b).map g = map₂ f' (a.map g') b := by cases a <;> cases b <;> simp [h_distrib]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- Symmetric statement to `Option.map_map₂_right_comm`. -/
theorem map_map₂_distrib_right {g : γ → δ} {f' : α → β' → δ} {g' : β → β'}
    (h_distrib : ∀ a b, g (f a b) = f' a (g' b)) : (map₂ f a b).map g = map₂ f' a (b.map g') := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    f : α → β → γ
    a : Option α
    b : Option β
    β' : Type u_6
    g : γ → δ
    f' : α → β' → δ
    g' : β → β'
    h_distrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' a (g' b))
    ⊢ Eq (Option.map g (Option.map₂ f a b)) (Option.map₂ f' a (Option.map g' b))
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
  cases a <;> cases b <;> simp [h_distrib]
                          /-
                            🎉 no goals
                          -/


/-- Symmetric statement to `Option.map_map₂_distrib_left`. -/
theorem map₂_map_left_comm {f : α' → β → γ} {g : α → α'} {f' : α → β → δ} {g' : δ → γ}
    (h_left_comm : ∀ a b, f (g a) b = g' (f' a b)) : map₂ f (a.map g) b = (map₂ f' a b).map g' := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    a : Option α
    b : Option β
    α' : Type u_5
    f : α' → β → γ
    g : α → α'
    f' : α → β → δ
    g' : δ → γ
    h_left_comm : ∀ (a : α) (b : β), Eq (f (g a) b) (g' (f' a b))
    ⊢ Eq (Option.map₂ f (Option.map g a) b) (Option.map g' (Option.map₂ f' a b))
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
  cases a <;> cases b <;> simp [h_left_comm]
                          /-
                            🎉 no goals
                          -/


/-- Symmetric statement to `Option.map_map₂_distrib_right`. -/
theorem map_map₂_right_comm {f : α → β' → γ} {g : β → β'} {f' : α → β → δ} {g' : δ → γ}
    (h_right_comm : ∀ a b, f a (g b) = g' (f' a b)) :
                                                    /-
                                                      α : Type u_1
                                                      β : Type u_2
                                                      γ : Type u_3
                                                      δ : Type u_4
                                                      a : Option α
                                                      b : Option β
                                                      β' : Type u_6
                                                      f : α → β' → γ
                                                      g : β → β'
                                                      f' : α → β → δ
                                                      g' : δ → γ
                                                      h_right_comm : ∀ (a : α) (b : β), Eq (f a (g b)) (g' (f' a b))
                                                      ⊢ Eq (Option.map₂ f a (Option.map g b)) (Option.map g' (Option.map₂ f' a b))
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
    map₂ f a (b.map g) = (map₂ f' a b).map g' := by cases a <;> cases b <;> simp [h_right_comm]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem map_map₂_antidistrib {g : γ → δ} {f' : β' → α' → δ} {g₁ : β → β'} {g₂ : α → α'}
    (h_antidistrib : ∀ a b, g (f a b) = f' (g₁ b) (g₂ a)) :
    (map₂ f a b).map g = map₂ f' (b.map g₁) (a.map g₂) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    f : α → β → γ
    a : Option α
    b : Option β
    α' : Type u_5
    β' : Type u_6
    g : γ → δ
    f' : β' → α' → δ
    g₁ : β → β'
    g₂ : α → α'
    h_antidistrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g₁ b) (g₂ a))
    ⊢ Eq (Option.map g (Option.map₂ f a b)) (Option.map₂ f' (Option.map g₁ b) (Opt …
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
  cases a <;> cases b <;> simp [h_antidistrib]
                          /-
                            🎉 no goals
                          -/


/-- Symmetric statement to `Option.map₂_map_left_anticomm`. -/
theorem map_map₂_antidistrib_left {g : γ → δ} {f' : β' → α → δ} {g' : β → β'}
    (h_antidistrib : ∀ a b, g (f a b) = f' (g' b) a) :
    (map₂ f a b).map g = map₂ f' (b.map g') a := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    f : α → β → γ
    a : Option α
    b : Option β
    β' : Type u_6
    g : γ → δ
    f' : β' → α → δ
    g' : β → β'
    h_antidistrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g' b) a)
    ⊢ Eq (Option.map g (Option.map₂ f a b)) (Option.map₂ f' (Option.map g' b) a)
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
  cases a <;> cases b <;> simp [h_antidistrib]
                          /-
                            🎉 no goals
                          -/


/-- Symmetric statement to `Option.map_map₂_right_anticomm`. -/
theorem map_map₂_antidistrib_right {g : γ → δ} {f' : β → α' → δ} {g' : α → α'}
    (h_antidistrib : ∀ a b, g (f a b) = f' b (g' a)) :
                                                    /-
                                                      α : Type u_1
                                                      β : Type u_2
                                                      γ : Type u_3
                                                      δ : Type u_4
                                                      f : α → β → γ
                                                      a : Option α
                                                      b : Option β
                                                      α' : Type u_5
                                                      g : γ → δ
                                                      f' : β → α' → δ
                                                      g' : α → α'
                                                      h_antidistrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' b (g' a))
                                                      ⊢ Eq (Option.map g (Option.map₂ f a b)) (Option.map₂ f' b (Option.map g' a))
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
    (map₂ f a b).map g = map₂ f' b (a.map g') := by cases a <;> cases b <;> simp [h_antidistrib]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- Symmetric statement to `Option.map_map₂_antidistrib_left`. -/
theorem map₂_map_left_anticomm {f : α' → β → γ} {g : α → α'} {f' : β → α → δ} {g' : δ → γ}
    (h_left_anticomm : ∀ a b, f (g a) b = g' (f' b a)) :
                                                    /-
                                                      α : Type u_1
                                                      β : Type u_2
                                                      γ : Type u_3
                                                      δ : Type u_4
                                                      a : Option α
                                                      b : Option β
                                                      α' : Type u_5
                                                      f : α' → β → γ
                                                      g : α → α'
                                                      f' : β → α → δ
                                                      g' : δ → γ
                                                      h_left_anticomm : ∀ (a : α) (b : β), Eq (f (g a) b) (g' (f' b a))
                                                      ⊢ Eq (Option.map₂ f (Option.map g a) b) (Option.map g' (Option.map₂ f' b a))
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
    map₂ f (a.map g) b = (map₂ f' b a).map g' := by cases a <;> cases b <;> simp [h_left_anticomm]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- Symmetric statement to `Option.map_map₂_antidistrib_right`. -/
theorem map_map₂_right_anticomm {f : α → β' → γ} {g : β → β'} {f' : β → α → δ} {g' : δ → γ}
    (h_right_anticomm : ∀ a b, f a (g b) = g' (f' b a)) :
                                                    /-
                                                      α : Type u_1
                                                      β : Type u_2
                                                      γ : Type u_3
                                                      δ : Type u_4
                                                      a : Option α
                                                      b : Option β
                                                      β' : Type u_6
                                                      f : α → β' → γ
                                                      g : β → β'
                                                      f' : β → α → δ
                                                      g' : δ → γ
                                                      h_right_anticomm : ∀ (a : α) (b : β), Eq (f a (g b)) (g' (f' b a))
                                                      ⊢ Eq (Option.map₂ f a (Option.map g b)) (Option.map g' (Option.map₂ f' b a))
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
    map₂ f a (b.map g) = (map₂ f' b a).map g' := by cases a <;> cases b <;> simp [h_right_anticomm]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- If `a` is a left identity for a binary operation `f`, then `some a` is a left identity for
`Option.map₂ f`. -/
lemma map₂_left_identity {f : α → β → β} {a : α} (h : ∀ b, f a b = b) (o : Option β) :
    map₂ f (some a) o = o := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β → β
    a : α
    h : ∀ (b : β), Eq (f a b) b
    o : Option β
    ⊢ Eq (Option.map₂ f (Option.some a) o) o
  -/
  cases o; exacts [rfl, congr_arg some (h _)]
           /-
             🎉 no goals
           -/


/-- If `b` is a right identity for a binary operation `f`, then `some b` is a right identity for
`Option.map₂ f`. -/
lemma map₂_right_identity {f : α → β → α} {b : β} (h : ∀ a, f a b = a) (o : Option α) :
    map₂ f o (some b) = o := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β → α
    b : β
    h : ∀ (a : α), Eq (f a b) a
    o : Option α
    ⊢ Eq (Option.map₂ f o (Option.some b)) o
  -/
  simp [h, map₂]
  /-
    🎉 no goals
  -/


