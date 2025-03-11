@[simp]
theorem mapAccumr_mapAccumr :
    mapAccumr f₁ (mapAccumr f₂ xs s₂).snd s₁
    = let m := (mapAccumr (fun x s =>
        let r₂ := f₂ x s.snd
        let r₁ := f₁ r₂.snd s.fst
        ((r₁.fst, r₂.fst), r₁.snd)
      ) xs (s₁, s₂))
      (m.fst.fst, m.snd) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ₁ : Type u_6
    σ₂ : Type u_7
    n : Nat
    s₁ : σ₁
    s₂ : σ₂
    xs : List.Vector α n
    f₁ : β → σ₁ → Prod σ₁ γ
    f₂ : α → σ₂ → Prod σ₂ β
    ⊢ Eq (List.Vector.mapAccumr f₁ (List.Vector.mapAccumr f₂ xs s₂).2 s₁)
        (let m :=
          List.Vector.mapAccumr
            (fun x s =>
              let r₂ := f₂ x s.2;
              let r₁ := f₁ r₂.2 s.1;
              { fst := { fst := r₁.1, snd := r₂.1 }, snd := r₁.2 })
            xs { fst := s₁, snd := s₂ };
        { fst := m.1.1, snd := m.2 })
  -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  induction xs using Vector.revInductionOn generalizing s₁ s₂ <;> simp_all
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem mapAccumr_map {s : σ₁} (f₂ : α → β) :
    (mapAccumr f₁ (map f₂ xs) s) = (mapAccumr (fun x s => f₁ (f₂ x) s) xs s) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ₁ : Type u_6
    n : Nat
    xs : List.Vector α n
    f₁ : β → σ₁ → Prod σ₁ γ
    s : σ₁
    f₂ : α → β
    ⊢ Eq (List.Vector.mapAccumr f₁ (List.Vector.map f₂ xs) s) (List.Vector.mapAccu …
  -/
                                                              /-
                                                                🎉 no goals
                                                              -/
  induction xs using Vector.revInductionOn generalizing s <;> simp_all
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem map_mapAccumr {s : σ₂} (f₁ : β → γ) :
    (map f₁ (mapAccumr f₂ xs s).snd) = (mapAccumr (fun x s =>
        let r := (f₂ x s); (r.fst, f₁ r.snd)
      ) xs s).snd := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ₂ : Type u_7
    n : Nat
    xs : List.Vector α n
    f₂ : α → σ₂ → Prod σ₂ β
    s : σ₂
    f₁ : β → γ
    ⊢ Eq (List.Vector.map f₁ (List.Vector.mapAccumr f₂ xs s).2)
        (List.Vector.mapAccumr
            (fun x s =>
              let r := f₂ x s;
              { fst := r.1, snd := f₁ r.2 })
            xs s).2
  -/
                                                              /-
                                                                🎉 no goals
                                                              -/
  induction xs using Vector.revInductionOn generalizing s <;> simp_all
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem map_map (f₁ : β → γ) (f₂ : α → β) :
    map f₁ (map f₂ xs) = map (fun x => f₁ <| f₂ x) xs := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    n : Nat
    xs : List.Vector α n
    f₁ : β → γ
    f₂ : α → β
    ⊢ Eq (List.Vector.map f₁ (List.Vector.map f₂ xs)) (List.Vector.map (fun x => f …
  -/
                   /-
                     🎉 no goals
                   -/
  induction xs <;> simp_all
                   /-
                     🎉 no goals
                   -/


theorem map_pmap {p : α → Prop} (f₁ : β → γ) (f₂ : (a : α) → p a → β) (H : ∀ x ∈ xs.toList, p x):
    map f₁ (pmap f₂ xs H) = pmap (fun x hx => f₁ <| f₂ x hx) xs H := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    n : Nat
    xs : List.Vector α n
    p : α → Prop
    f₁ : β → γ
    f₂ : (a : α) → p a → β
    H : ∀ (x : α), Membership.mem xs.toList x → p x
    ⊢ Eq (List.Vector.map f₁ (List.Vector.pmap f₂ xs H)) (List.Vector.pmap (fun x  …
  -/
                   /-
                     🎉 no goals
                   -/
  induction xs <;> simp_all
                   /-
                     🎉 no goals
                   -/


theorem pmap_map {p : β → Prop} (f₁ : (b : β) → p b → γ) (f₂ : α → β)
    (H : ∀ x ∈ (xs.map f₂).toList, p x):
                                                                   /-
                                                                     α : Type u_1
                                                                     β : Type u_2
                                                                     γ : Type u_3
                                                                     ζ : Type u_4
                                                                     σ : Type u_5
                                                                     σ₁ : Type u_6
                                                                     σ₂ : Type u_7
                                                                     φ : Type u_8
                                                                     n : Nat
                                                                     s : σ
                                                                     s₁ : σ₁
                                                                     s₂ : σ₂
                                                                     xs : List.Vector α n
                                                                     f₁✝ : β → σ₁ → Prod σ₁ γ
                                                                     f₂✝ : α → σ₂ → Prod σ₂ β
                                                                     p : β → Prop
                                                                     f₁ : (b : β) → p b → γ
                                                                     f₂ : α → β
                                                                     H : ∀ (x : β), Membership.mem (List.Vector.map f₂ xs).toList x → p x
                                                                     ⊢ ∀ (x : α), Membership.mem xs.toList x → p (f₂ x)
                                                                   -/
    pmap f₁ (map f₂ xs) H = pmap (fun x hx => f₁ (f₂ x) hx) xs (by simpa using H) := by
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    n : Nat
    xs : List.Vector α n
    p : β → Prop
    f₁ : (b : β) → p b → γ
    f₂ : α → β
    H : ∀ (x : β), Membership.mem (List.Vector.map f₂ xs).toList x → p x
    ⊢ Eq (List.Vector.pmap f₁ (List.Vector.map f₂ xs) H) (List.Vector.pmap (fun x  …
  -/
                   /-
                     🎉 no goals
                   -/
  induction xs <;> simp_all
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem mapAccumr₂_mapAccumr_left (f₁ : γ → β → σ₁ → σ₁ × ζ) (f₂ : α → σ₂ → σ₂ × γ) :
    (mapAccumr₂ f₁ (mapAccumr f₂ xs s₂).snd ys s₁)
    = let m := (mapAccumr₂ (fun x y s =>
          let r₂ := f₂ x s.snd
          let r₁ := f₁ r₂.snd y s.fst
          ((r₁.fst, r₂.fst), r₁.snd)
        ) xs ys (s₁, s₂))
      (m.fst.fst, m.snd) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ζ : Type u_4
    σ₁ : Type u_6
    σ₂ : Type u_7
    n : Nat
    s₁ : σ₁
    s₂ : σ₂
    xs : List.Vector α n
    ys : List.Vector β n
    f₁ : γ → β → σ₁ → Prod σ₁ ζ
    f₂ : α → σ₂ → Prod σ₂ γ
    ⊢ Eq (List.Vector.mapAccumr₂ f₁ (List.Vector.mapAccumr f₂ xs s₂).2 ys s₁)
        (let m :=
          List.Vector.mapAccumr₂
            (fun x y s =>
              let r₂ := f₂ x s.2;
              let r₁ := f₁ r₂.2 y s.1;
              { fst := { fst := r₁.1, snd := r₂.1 }, snd := r₁.2 })
            xs ys { fst := s₁, snd := s₂ };
        { fst := m.1.1, snd := m.2 })
  -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  induction xs, ys using Vector.revInductionOn₂ generalizing s₁ s₂ <;> simp_all
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem map₂_map_left (f₁ : γ → β → ζ) (f₂ : α → γ) :
    map₂ f₁ (map f₂ xs) ys = map₂ (fun x y => f₁ (f₂ x) y) xs ys := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ζ : Type u_4
    n : Nat
    xs : List.Vector α n
    ys : List.Vector β n
    f₁ : γ → β → ζ
    f₂ : α → γ
    ⊢ Eq (List.Vector.map₂ f₁ (List.Vector.map f₂ xs) ys) (List.Vector.map₂ (fun x …
  -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  induction xs, ys using Vector.revInductionOn₂ <;> simp_all
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem mapAccumr₂_mapAccumr_right (f₁ : α → γ → σ₁ → σ₁ × ζ) (f₂ : β → σ₂ → σ₂ × γ) :
    (mapAccumr₂ f₁ xs (mapAccumr f₂ ys s₂).snd s₁)
    = let m := (mapAccumr₂ (fun x y s =>
          let r₂ := f₂ y s.snd
          let r₁ := f₁ x r₂.snd s.fst
          ((r₁.fst, r₂.fst), r₁.snd)
        ) xs ys (s₁, s₂))
      (m.fst.fst, m.snd) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ζ : Type u_4
    σ₁ : Type u_6
    σ₂ : Type u_7
    n : Nat
    s₁ : σ₁
    s₂ : σ₂
    xs : List.Vector α n
    ys : List.Vector β n
    f₁ : α → γ → σ₁ → Prod σ₁ ζ
    f₂ : β → σ₂ → Prod σ₂ γ
    ⊢ Eq (List.Vector.mapAccumr₂ f₁ xs (List.Vector.mapAccumr f₂ ys s₂).2 s₁)
        (let m :=
          List.Vector.mapAccumr₂
            (fun x y s =>
              let r₂ := f₂ y s.2;
              let r₁ := f₁ x r₂.2 s.1;
              { fst := { fst := r₁.1, snd := r₂.1 }, snd := r₁.2 })
            xs ys { fst := s₁, snd := s₂ };
        { fst := m.1.1, snd := m.2 })
  -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  induction xs, ys using Vector.revInductionOn₂ generalizing s₁ s₂ <;> simp_all
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem map₂_map_right (f₁ : α → γ → ζ) (f₂ : β → γ) :
    map₂ f₁ xs (map f₂ ys) = map₂ (fun x y => f₁ x (f₂ y)) xs ys := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ζ : Type u_4
    n : Nat
    xs : List.Vector α n
    ys : List.Vector β n
    f₁ : α → γ → ζ
    f₂ : β → γ
    ⊢ Eq (List.Vector.map₂ f₁ xs (List.Vector.map f₂ ys)) (List.Vector.map₂ (fun x …
  -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  induction xs, ys using Vector.revInductionOn₂ <;> simp_all
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem mapAccumr_mapAccumr₂ (f₁ : γ → σ₁ → σ₁ × ζ) (f₂ : α → β → σ₂ → σ₂ × γ) :
    (mapAccumr f₁ (mapAccumr₂ f₂ xs ys s₂).snd s₁)
    = let m := mapAccumr₂ (fun x y s =>
          let r₂ := f₂ x y s.snd
          let r₁ := f₁ r₂.snd s.fst
          ((r₁.fst, r₂.fst), r₁.snd)
        ) xs ys (s₁, s₂)
      (m.fst.fst, m.snd) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ζ : Type u_4
    σ₁ : Type u_6
    σ₂ : Type u_7
    n : Nat
    s₁ : σ₁
    s₂ : σ₂
    xs : List.Vector α n
    ys : List.Vector β n
    f₁ : γ → σ₁ → Prod σ₁ ζ
    f₂ : α → β → σ₂ → Prod σ₂ γ
    ⊢ Eq (List.Vector.mapAccumr f₁ (List.Vector.mapAccumr₂ f₂ xs ys s₂).2 s₁)
        (let m :=
          List.Vector.mapAccumr₂
            (fun x y s =>
              let r₂ := f₂ x y s.2;
              let r₁ := f₁ r₂.2 s.1;
              { fst := { fst := r₁.1, snd := r₂.1 }, snd := r₁.2 })
            xs ys { fst := s₁, snd := s₂ };
        { fst := m.1.1, snd := m.2 })
  -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  induction xs, ys using Vector.revInductionOn₂ generalizing s₁ s₂ <;> simp_all
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem map_map₂ (f₁ : γ → ζ) (f₂ : α → β → γ) :
    map f₁ (map₂ f₂ xs ys) = map₂ (fun x y => f₁ <| f₂ x y) xs ys := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ζ : Type u_4
    n : Nat
    xs : List.Vector α n
    ys : List.Vector β n
    f₁ : γ → ζ
    f₂ : α → β → γ
    ⊢ Eq (List.Vector.map f₁ (List.Vector.map₂ f₂ xs ys)) (List.Vector.map₂ (fun x …
  -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  induction xs, ys using Vector.revInductionOn₂ <;> simp_all
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem mapAccumr₂_mapAccumr₂_left_left (f₁ : γ → α → σ₁ → σ₁ × φ) (f₂ : α → β → σ₂ → σ₂ × γ) :
    (mapAccumr₂ f₁ (mapAccumr₂ f₂ xs ys s₂).snd xs s₁)
    = let m := mapAccumr₂ (fun x y (s₁, s₂) =>
                let r₂ := f₂ x y s₂
                let r₁ := f₁ r₂.snd x s₁
                ((r₁.fst, r₂.fst), r₁.snd)
              )
            xs ys (s₁, s₂)
    (m.fst.fst, m.snd) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ₁ : Type u_6
    σ₂ : Type u_7
    φ : Type u_8
    n : Nat
    s₁ : σ₁
    s₂ : σ₂
    xs : List.Vector α n
    ys : List.Vector β n
    f₁ : γ → α → σ₁ → Prod σ₁ φ
    f₂ : α → β → σ₂ → Prod σ₂ γ
    ⊢ Eq (List.Vector.mapAccumr₂ f₁ (List.Vector.mapAccumr₂ f₂ xs ys s₂).2 xs s₁)
        (let m :=
          List.Vector.mapAccumr₂
            (fun x y x_1 =>
              List.Vector.mapAccumr₂_mapAccumr₂_left_left.match_1 (fun x => Prod ( …
                let r₂ := f₂ x y s₂;
                let r₁ := f₁ r₂.2 x s₁;
                { fst := { fst := r₁.1, snd := r₂.1 }, snd := r₁.2 })
            xs ys { fst := s₁, snd := s₂ };
        { fst := m.1.1, snd := m.2 })
  -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  induction xs, ys using Vector.revInductionOn₂ generalizing s₁ s₂ <;> simp_all
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem mapAccumr₂_mapAccumr₂_left_right
    (f₁ : γ → β → σ₁ → σ₁ × φ) (f₂ : α → β → σ₂ → σ₂ × γ) :
    (mapAccumr₂ f₁ (mapAccumr₂ f₂ xs ys s₂).snd ys s₁)
    = let m := mapAccumr₂ (fun x y (s₁, s₂) =>
                let r₂ := f₂ x y s₂
                let r₁ := f₁ r₂.snd y s₁
                ((r₁.fst, r₂.fst), r₁.snd)
              )
            xs ys (s₁, s₂)
    (m.fst.fst, m.snd) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ₁ : Type u_6
    σ₂ : Type u_7
    φ : Type u_8
    n : Nat
    s₁ : σ₁
    s₂ : σ₂
    xs : List.Vector α n
    ys : List.Vector β n
    f₁ : γ → β → σ₁ → Prod σ₁ φ
    f₂ : α → β → σ₂ → Prod σ₂ γ
    ⊢ Eq (List.Vector.mapAccumr₂ f₁ (List.Vector.mapAccumr₂ f₂ xs ys s₂).2 ys s₁)
        (let m :=
          List.Vector.mapAccumr₂
            (fun x y x_1 =>
              List.Vector.mapAccumr₂_mapAccumr₂_left_left.match_1 (fun x => Prod ( …
                let r₂ := f₂ x y s₂;
                let r₁ := f₁ r₂.2 y s₁;
                { fst := { fst := r₁.1, snd := r₂.1 }, snd := r₁.2 })
            xs ys { fst := s₁, snd := s₂ };
        { fst := m.1.1, snd := m.2 })
  -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  induction xs, ys using Vector.revInductionOn₂ generalizing s₁ s₂ <;> simp_all
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem mapAccumr₂_mapAccumr₂_right_left (f₁ : α → γ → σ₁ → σ₁ × φ) (f₂ : α → β → σ₂ → σ₂ × γ) :
    (mapAccumr₂ f₁ xs (mapAccumr₂ f₂ xs ys s₂).snd s₁)
    = let m := mapAccumr₂ (fun x y (s₁, s₂) =>
                let r₂ := f₂ x y s₂
                let r₁ := f₁ x r₂.snd s₁
                ((r₁.fst, r₂.fst), r₁.snd)
              )
            xs ys (s₁, s₂)
    (m.fst.fst, m.snd) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ₁ : Type u_6
    σ₂ : Type u_7
    φ : Type u_8
    n : Nat
    s₁ : σ₁
    s₂ : σ₂
    xs : List.Vector α n
    ys : List.Vector β n
    f₁ : α → γ → σ₁ → Prod σ₁ φ
    f₂ : α → β → σ₂ → Prod σ₂ γ
    ⊢ Eq (List.Vector.mapAccumr₂ f₁ xs (List.Vector.mapAccumr₂ f₂ xs ys s₂).2 s₁)
        (let m :=
          List.Vector.mapAccumr₂
            (fun x y x_1 =>
              List.Vector.mapAccumr₂_mapAccumr₂_left_left.match_1 (fun x => Prod ( …
                let r₂ := f₂ x y s₂;
                let r₁ := f₁ x r₂.2 s₁;
                { fst := { fst := r₁.1, snd := r₂.1 }, snd := r₁.2 })
            xs ys { fst := s₁, snd := s₂ };
        { fst := m.1.1, snd := m.2 })
  -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  induction xs, ys using Vector.revInductionOn₂ generalizing s₁ s₂ <;> simp_all
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem mapAccumr₂_mapAccumr₂_right_right (f₁ : β → γ → σ₁ → σ₁ × φ) (f₂ : α → β → σ₂ → σ₂ × γ) :
    (mapAccumr₂ f₁ ys (mapAccumr₂ f₂ xs ys s₂).snd s₁)
    = let m := mapAccumr₂ (fun x y (s₁, s₂) =>
                let r₂ := f₂ x y s₂
                let r₁ := f₁ y r₂.snd s₁
                ((r₁.fst, r₂.fst), r₁.snd)
              )
            xs ys (s₁, s₂)
    (m.fst.fst, m.snd) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ₁ : Type u_6
    σ₂ : Type u_7
    φ : Type u_8
    n : Nat
    s₁ : σ₁
    s₂ : σ₂
    xs : List.Vector α n
    ys : List.Vector β n
    f₁ : β → γ → σ₁ → Prod σ₁ φ
    f₂ : α → β → σ₂ → Prod σ₂ γ
    ⊢ Eq (List.Vector.mapAccumr₂ f₁ ys (List.Vector.mapAccumr₂ f₂ xs ys s₂).2 s₁)
        (let m :=
          List.Vector.mapAccumr₂
            (fun x y x_1 =>
              List.Vector.mapAccumr₂_mapAccumr₂_left_left.match_1 (fun x => Prod ( …
                let r₂ := f₂ x y s₂;
                let r₁ := f₁ y r₂.2 s₁;
                { fst := { fst := r₁.1, snd := r₂.1 }, snd := r₁.2 })
            xs ys { fst := s₁, snd := s₂ };
        { fst := m.1.1, snd := m.2 })
  -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  induction xs, ys using Vector.revInductionOn₂ generalizing s₁ s₂ <;> simp_all
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem mapAccumr_bisim {f₁ : α → σ₁ → σ₁ × β} {f₂ : α → σ₂ → σ₂ × β} {s₁ : σ₁} {s₂ : σ₂}
    (R : σ₁ → σ₂ → Prop) (h₀ : R s₁ s₂)
    (hR : ∀ {s q} a, R s q → R (f₁ a s).1 (f₂ a q).1 ∧ (f₁ a s).2 = (f₂ a q).2) :
    R (mapAccumr f₁ xs s₁).fst (mapAccumr f₂ xs s₂).fst
    ∧ (mapAccumr f₁ xs s₁).snd = (mapAccumr f₂ xs s₂).snd := by
  /-
    α : Type u_1
    β : Type u_2
    σ₁ : Type u_6
    σ₂ : Type u_7
    n : Nat
    xs : List.Vector α n
    f₁ : α → σ₁ → Prod σ₁ β
    f₂ : α → σ₂ → Prod σ₂ β
    s₁ : σ₁
    s₂ : σ₂
    R : σ₁ → σ₂ → Prop
    h₀ : R s₁ s₂
    hR : ∀ {s : σ₁} {q : σ₂} (a : α), R s q → And (R (f₁ a s).1 (f₂ a q).1) (Eq (f …
    ⊢ And (R (List.Vector.mapAccumr f₁ xs s₁).1 (List.Vector.mapAccumr f₂ xs s₂).1 …
  -/
  induction xs using Vector.revInductionOn generalizing s₁ s₂
  /-
    case nil
    α : Type u_1
    β : Type u_2
    σ₁ : Type u_6
    σ₂ : Type u_7
    n : Nat
    xs : List.Vector α n
    f₁ : α → σ₁ → Prod σ₁ β
    f₂ : α → σ₂ → Prod σ₂ β
    R : σ₁ → σ₂ → Prop
    hR : ∀ {s : σ₁} {q : σ₂} (a : α), R s q → And (R (f₁ a s).1 (f₂ a q).1) (Eq (f …
    s₁ : σ₁
    s₂ : σ₂
    h₀ : R s₁ s₂
    ⊢ And (R (List.Vector.mapAccumr f₁ List.Vector.nil s₁).1 (List.Vector.mapAccum …
  -/
  next => exact ⟨h₀, rfl⟩
  next xs x ih =>
    rcases (hR x h₀) with ⟨hR, _⟩
    simp only [mapAccumr_snoc, ih hR, true_and]
    congr 1


theorem mapAccumr_bisim_tail {f₁ : α → σ₁ → σ₁ × β} {f₂ : α → σ₂ → σ₂ × β} {s₁ : σ₁} {s₂ : σ₂}
    (h : ∃ R : σ₁ → σ₂ → Prop, R s₁ s₂ ∧
      ∀ {s q} a, R s q → R (f₁ a s).1 (f₂ a q).1 ∧ (f₁ a s).2 = (f₂ a q).2) :
    (mapAccumr f₁ xs s₁).snd = (mapAccumr f₂ xs s₂).snd := by
  /-
    α : Type u_1
    β : Type u_2
    σ₁ : Type u_6
    σ₂ : Type u_7
    n : Nat
    xs : List.Vector α n
    f₁ : α → σ₁ → Prod σ₁ β
    f₂ : α → σ₂ → Prod σ₂ β
    s₁ : σ₁
    s₂ : σ₂
    h : Exists fun R => And (R s₁ s₂) (∀ {s : σ₁} {q : σ₂} (a : α), R s q → And (R …
    ⊢ Eq (List.Vector.mapAccumr f₁ xs s₁).2 (List.Vector.mapAccumr f₂ xs s₂).2
  -/
  rcases h with ⟨R, h₀, hR⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    σ₁ : Type u_6
    σ₂ : Type u_7
    n : Nat
    xs : List.Vector α n
    f₁ : α → σ₁ → Prod σ₁ β
    f₂ : α → σ₂ → Prod σ₂ β
    s₁ : σ₁
    s₂ : σ₂
    R : σ₁ → σ₂ → Prop
    h₀ : R s₁ s₂
    hR : ∀ {s : σ₁} {q : σ₂} (a : α), R s q → And (R (f₁ a s).1 (f₂ a q).1) (Eq (f …
    ⊢ Eq (List.Vector.mapAccumr f₁ xs s₁).2 (List.Vector.mapAccumr f₂ xs s₂).2
  -/
  exact (mapAccumr_bisim R h₀ hR).2
  /-
    🎉 no goals
  -/


theorem mapAccumr₂_bisim {ys : Vector β n} {f₁ : α → β → σ₁ → σ₁ × γ}
    {f₂ : α → β → σ₂ → σ₂ × γ} {s₁ : σ₁} {s₂ : σ₂}
    (R : σ₁ → σ₂ → Prop) (h₀ : R s₁ s₂)
    (hR :  ∀ {s q} a b, R s q → R (f₁ a b s).1 (f₂ a b q).1 ∧ (f₁ a b s).2 = (f₂ a b q).2) :
    R (mapAccumr₂ f₁ xs ys s₁).1 (mapAccumr₂ f₂ xs ys s₂).1
    ∧ (mapAccumr₂ f₁ xs ys s₁).2 = (mapAccumr₂ f₂ xs ys s₂).2 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ₁ : Type u_6
    σ₂ : Type u_7
    n : Nat
    xs : List.Vector α n
    ys : List.Vector β n
    f₁ : α → β → σ₁ → Prod σ₁ γ
    f₂ : α → β → σ₂ → Prod σ₂ γ
    s₁ : σ₁
    s₂ : σ₂
    R : σ₁ → σ₂ → Prop
    h₀ : R s₁ s₂
    hR : ∀ {s : σ₁} {q : σ₂} (a : α) (b : β), R s q → And (R (f₁ a b s).1 (f₂ a b  …
    ⊢ And (R (List.Vector.mapAccumr₂ f₁ xs ys s₁).1 (List.Vector.mapAccumr₂ f₂ xs  …
  -/
  induction xs, ys using Vector.revInductionOn₂ generalizing s₁ s₂
  /-
    case nil
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ₁ : Type u_6
    σ₂ : Type u_7
    n : Nat
    xs : List.Vector α n
    f₁ : α → β → σ₁ → Prod σ₁ γ
    f₂ : α → β → σ₂ → Prod σ₂ γ
    R : σ₁ → σ₂ → Prop
    hR : ∀ {s : σ₁} {q : σ₂} (a : α) (b : β), R s q → And (R (f₁ a b s).1 (f₂ a b  …
    s₁ : σ₁
    s₂ : σ₂
    h₀ : R s₁ s₂
    ⊢ And (R (List.Vector.mapAccumr₂ f₁ List.Vector.nil List.Vector.nil s₁).1 (Lis …
  -/
  next => exact ⟨h₀, rfl⟩
  next xs ys x y ih =>
    rcases (hR x y h₀) with ⟨hR, _⟩
    simp only [mapAccumr₂_snoc, ih hR, true_and]
    congr 1


theorem mapAccumr₂_bisim_tail {ys : Vector β n} {f₁ : α → β → σ₁ → σ₁ × γ}
    {f₂ : α → β → σ₂ → σ₂ × γ} {s₁ : σ₁} {s₂ : σ₂}
    (h : ∃ R : σ₁ → σ₂ → Prop, R s₁ s₂ ∧
      ∀ {s q} a b, R s q → R (f₁ a b s).1 (f₂ a b q).1 ∧ (f₁ a b s).2 = (f₂ a b q).2) :
    (mapAccumr₂ f₁ xs ys s₁).2 = (mapAccumr₂ f₂ xs ys s₂).2 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ₁ : Type u_6
    σ₂ : Type u_7
    n : Nat
    xs : List.Vector α n
    ys : List.Vector β n
    f₁ : α → β → σ₁ → Prod σ₁ γ
    f₂ : α → β → σ₂ → Prod σ₂ γ
    s₁ : σ₁
    s₂ : σ₂
    h : Exists fun R => And (R s₁ s₂) (∀ {s : σ₁} {q : σ₂} (a : α) (b : β), R s q  …
    ⊢ Eq (List.Vector.mapAccumr₂ f₁ xs ys s₁).2 (List.Vector.mapAccumr₂ f₂ xs ys s …
  -/
  rcases h with ⟨R, h₀, hR⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ₁ : Type u_6
    σ₂ : Type u_7
    n : Nat
    xs : List.Vector α n
    ys : List.Vector β n
    f₁ : α → β → σ₁ → Prod σ₁ γ
    f₂ : α → β → σ₂ → Prod σ₂ γ
    s₁ : σ₁
    s₂ : σ₂
    R : σ₁ → σ₂ → Prop
    h₀ : R s₁ s₂
    hR : ∀ {s : σ₁} {q : σ₂} (a : α) (b : β), R s q → And (R (f₁ a b s).1 (f₂ a b  …
    ⊢ Eq (List.Vector.mapAccumr₂ f₁ xs ys s₁).2 (List.Vector.mapAccumr₂ f₂ xs ys s …
  -/
  exact (mapAccumr₂_bisim R h₀ hR).2
  /-
    🎉 no goals
  -/


protected theorem map_eq_mapAccumr {f : α → β} :
    map f xs = (mapAccumr (fun x (_ : Unit) ↦ ((), f x)) xs ()).snd := by
  /-
    α : Type u_1
    β : Type u_2
    n : Nat
    xs : List.Vector α n
    f : α → β
    ⊢ Eq (List.Vector.map f xs) (List.Vector.mapAccumr (fun x x_1 => { fst := Unit …
  -/
                                               /-
                                                 🎉 no goals
                                               -/
  induction xs using Vector.revInductionOn <;> simp_all
                                               /-
                                                 🎉 no goals
                                               -/


/--
  If there is a set of states that is closed under `f`, and such that `f` produces that same output
  for all states in this set, then the state is not actually needed.
  Hence, then we can rewrite `mapAccumr` into just `map`
-/
theorem mapAccumr_eq_map {f : α → σ → σ × β} {s₀ : σ} (S : Set σ) (h₀ : s₀ ∈ S)
    (closure : ∀ a s, s ∈ S → (f a s).1 ∈ S)
    (out : ∀ a s s', s ∈ S → s' ∈ S → (f a s).2 = (f a s').2) :
    (mapAccumr f xs s₀).snd = map (f · s₀ |>.snd) xs := by
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_5
    n : Nat
    xs : List.Vector α n
    f : α → σ → Prod σ β
    s₀ : σ
    S : Set σ
    h₀ : Membership.mem S s₀
    closure : ∀ (a : α) (s : σ), Membership.mem S s → Membership.mem S (f a s).1
    out : ∀ (a : α) (s s' : σ), Membership.mem S s → Membership.mem S s' → Eq (f a …
    ⊢ Eq (List.Vector.mapAccumr f xs s₀).2 (List.Vector.map (fun x => (f x s₀).2)  …
  -/
  rw [Vector.map_eq_mapAccumr]
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_5
    n : Nat
    xs : List.Vector α n
    f : α → σ → Prod σ β
    s₀ : σ
    S : Set σ
    h₀ : Membership.mem S s₀
    closure : ∀ (a : α) (s : σ), Membership.mem S s → Membership.mem S (f a s).1
    out : ∀ (a : α) (s s' : σ), Membership.mem S s → Membership.mem S s' → Eq (f a …
    ⊢ Eq (List.Vector.mapAccumr f xs s₀).2 (List.Vector.mapAccumr (fun x x_1 => {  …
  -/
  apply mapAccumr_bisim_tail
  /-
    case h
    α : Type u_1
    β : Type u_2
    σ : Type u_5
    n : Nat
    xs : List.Vector α n
    f : α → σ → Prod σ β
    s₀ : σ
    S : Set σ
    h₀ : Membership.mem S s₀
    closure : ∀ (a : α) (s : σ), Membership.mem S s → Membership.mem S (f a s).1
    out : ∀ (a : α) (s s' : σ), Membership.mem S s → Membership.mem S s' → Eq (f a …
    ⊢ Exists fun R => And (R s₀ Unit.unit) (∀ {s : σ} {q : Unit} (a : α), R s q →  …
  -/
  use fun s _ => s ∈ S, h₀
  /-
    case right
    α : Type u_1
    β : Type u_2
    σ : Type u_5
    n : Nat
    xs : List.Vector α n
    f : α → σ → Prod σ β
    s₀ : σ
    S : Set σ
    h₀ : Membership.mem S s₀
    closure : ∀ (a : α) (s : σ), Membership.mem S s → Membership.mem S (f a s).1
    out : ∀ (a : α) (s s' : σ), Membership.mem S s → Membership.mem S s' → Eq (f a …
    ⊢ ∀ {s : σ} {q : Unit} (a : α), Membership.mem S s → And (Membership.mem S (f  …
  -/
  exact @fun s _q a h => ⟨closure a s h, out a s s₀ h h₀⟩
  /-
    🎉 no goals
  -/


protected theorem map₂_eq_mapAccumr₂ {f : α → β → γ} :
    map₂ f xs ys = (mapAccumr₂ (fun x y (_ : Unit) ↦ ((), f x y)) xs ys ()).snd := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    n : Nat
    xs : List.Vector α n
    ys : List.Vector β n
    f : α → β → γ
    ⊢ Eq (List.Vector.map₂ f xs ys) (List.Vector.mapAccumr₂ (fun x y x_1 => { fst  …
  -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  induction xs, ys using Vector.revInductionOn₂ <;> simp_all
                                                    /-
                                                      🎉 no goals
                                                    -/


/--
  If there is a set of states that is closed under `f`, and such that `f` produces that same output
  for all states in this set, then the state is not actually needed.
  Hence, then we can rewrite `mapAccumr₂` into just `map₂`
-/
theorem mapAccumr₂_eq_map₂ {f : α → β → σ → σ × γ} {s₀ : σ} (S : Set σ) (h₀ : s₀ ∈ S)
    (closure : ∀ a b s, s ∈ S → (f a b s).1 ∈ S)
    (out : ∀ a b s s', s ∈ S → s' ∈ S → (f a b s).2 = (f a b s').2) :
    (mapAccumr₂ f xs ys s₀).snd = map₂ (f · · s₀ |>.snd) xs ys := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ : Type u_5
    n : Nat
    xs : List.Vector α n
    ys : List.Vector β n
    f : α → β → σ → Prod σ γ
    s₀ : σ
    S : Set σ
    h₀ : Membership.mem S s₀
    closure : ∀ (a : α) (b : β) (s : σ), Membership.mem S s → Membership.mem S (f  …
    out : ∀ (a : α) (b : β) (s s' : σ), Membership.mem S s → Membership.mem S s' → …
    ⊢ Eq (List.Vector.mapAccumr₂ f xs ys s₀).2 (List.Vector.map₂ (fun x1 x2 => (f  …
  -/
  rw [Vector.map₂_eq_mapAccumr₂]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ : Type u_5
    n : Nat
    xs : List.Vector α n
    ys : List.Vector β n
    f : α → β → σ → Prod σ γ
    s₀ : σ
    S : Set σ
    h₀ : Membership.mem S s₀
    closure : ∀ (a : α) (b : β) (s : σ), Membership.mem S s → Membership.mem S (f  …
    out : ∀ (a : α) (b : β) (s s' : σ), Membership.mem S s → Membership.mem S s' → …
    ⊢ Eq (List.Vector.mapAccumr₂ f xs ys s₀).2 (List.Vector.mapAccumr₂ (fun x y x_ …
  -/
  apply mapAccumr₂_bisim_tail
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ : Type u_5
    n : Nat
    xs : List.Vector α n
    ys : List.Vector β n
    f : α → β → σ → Prod σ γ
    s₀ : σ
    S : Set σ
    h₀ : Membership.mem S s₀
    closure : ∀ (a : α) (b : β) (s : σ), Membership.mem S s → Membership.mem S (f  …
    out : ∀ (a : α) (b : β) (s s' : σ), Membership.mem S s → Membership.mem S s' → …
    ⊢ Exists fun R => And (R s₀ Unit.unit) (∀ {s : σ} {q : Unit} (a : α) (b : β),  …
  -/
  use fun s _ => s ∈ S, h₀
  /-
    case right
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ : Type u_5
    n : Nat
    xs : List.Vector α n
    ys : List.Vector β n
    f : α → β → σ → Prod σ γ
    s₀ : σ
    S : Set σ
    h₀ : Membership.mem S s₀
    closure : ∀ (a : α) (b : β) (s : σ), Membership.mem S s → Membership.mem S (f  …
    out : ∀ (a : α) (b : β) (s s' : σ), Membership.mem S s → Membership.mem S s' → …
    ⊢ ∀ {s : σ} {q : Unit} (a : α) (b : β), Membership.mem S s → And (Membership.m …
  -/
  exact @fun s _q a b h => ⟨closure a b s h, out a b s s₀ h h₀⟩
  /-
    🎉 no goals
  -/


/--
  If an accumulation function `f`, given an initial state `s`, produces `s` as its output state
  for all possible input bits, then the state is redundant and can be optimized out
-/
@[simp]
theorem mapAccumr_eq_map_of_constant_state (f : α → σ → σ × β) (s : σ) (h : ∀ a, (f a s).fst = s) :
    mapAccumr f xs s = (s, (map (fun x => (f x s).snd) xs)) := by
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_5
    n : Nat
    xs : List.Vector α n
    f : α → σ → Prod σ β
    s : σ
    h : ∀ (a : α), Eq (f a s).1 s
    ⊢ Eq (List.Vector.mapAccumr f xs s) { fst := s, snd := List.Vector.map (fun x  …
  -/
                                        /-
                                          🎉 no goals
                                        -/
  induction xs using revInductionOn <;> simp_all
                                        /-
                                          🎉 no goals
                                        -/


/--
  If an accumulation function `f`, given an initial state `s`, produces `s` as its output state
  for all possible input bits, then the state is redundant and can be optimized out
-/
@[simp]
theorem mapAccumr₂_eq_map₂_of_constant_state (f : α → β → σ → σ × γ) (s : σ)
    (h : ∀ a b, (f a b s).fst = s) :
    mapAccumr₂ f xs ys s = (s, (map₂ (fun x y => (f x y s).snd) xs ys)) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ : Type u_5
    n : Nat
    xs : List.Vector α n
    ys : List.Vector β n
    f : α → β → σ → Prod σ γ
    s : σ
    h : ∀ (a : α) (b : β), Eq (f a b s).1 s
    ⊢ Eq (List.Vector.mapAccumr₂ f xs ys s) { fst := s, snd := List.Vector.map₂ (f …
  -/
                                             /-
                                               🎉 no goals
                                             -/
  induction xs, ys using revInductionOn₂ <;> simp_all
                                             /-
                                               🎉 no goals
                                             -/


/--
  If an accumulation function `f`, produces the same output bits regardless of accumulation state,
  then the state is redundant and can be optimized out
-/
@[simp]
theorem mapAccumr_eq_map_of_unused_state (f : α → σ → σ × β) (s : σ)
    (h : ∀ a s s', (f a s).snd = (f a s').snd) :
    (mapAccumr f xs s).snd = (map (fun x => (f x s).snd) xs) :=
  mapAccumr_eq_map (fun _ => true) rfl (fun _ _ _ => rfl) (fun a s s' _ _ => h a s s')



/--
  If an accumulation function `f`, produces the same output bits regardless of accumulation state,
  then the state is redundant and can be optimized out
-/
@[simp]
theorem mapAccumr₂_eq_map₂_of_unused_state (f : α → β → σ → σ × γ) (s : σ)
    (h : ∀ a b s s', (f a b s).snd = (f a b s').snd) :
    (mapAccumr₂ f xs ys s).snd = (map₂ (fun x y => (f x y s).snd) xs ys) :=
  mapAccumr₂_eq_map₂ (fun _ => true) rfl (fun _ _ _ _ => rfl) (fun a b s s' _ _ => h a b s s')



/-- If `f` takes a pair of states, but always returns the same value for both elements of the
    pair, then we can simplify to just a single element of state
  -/
@[simp]
theorem mapAccumr_redundant_pair (f : α → (σ × σ) → (σ × σ) × β)
    (h : ∀ x s, (f x (s, s)).fst.fst = (f x (s, s)).fst.snd) :
    (mapAccumr f xs (s, s)).snd = (mapAccumr (fun x (s : σ) =>
      (f x (s, s) |>.fst.fst, f x (s, s) |>.snd)
    ) xs s).snd :=
  mapAccumr_bisim_tail <| by
    /-
      α : Type u_1
      β : Type u_2
      σ : Type u_5
      n : Nat
      s : σ
      xs : List.Vector α n
      f : α → Prod σ σ → Prod (Prod σ σ) β
      h : ∀ (x : α) (s : σ), Eq (f x { fst := s, snd := s }).1.1 (f x { fst := s, sn …
      ⊢ Exists fun R => And (R { fst := s, snd := s } s) (∀ {s : Prod σ σ} {q : σ} ( …
    -/
    use fun (s₁, s₂) s => s₂ = s ∧ s₁ = s
    /-
      case h
      α : Type u_1
      β : Type u_2
      σ : Type u_5
      n : Nat
      s : σ
      xs : List.Vector α n
      f : α → Prod σ σ → Prod (Prod σ σ) β
      h : ∀ (x : α) (s : σ), Eq (f x { fst := s, snd := s }).1.1 (f x { fst := s, sn …
      ⊢ And ((fun x s => List.Vector.mapAccumr_redundant_pair.match_1 (fun x => Prop …
    -/
    simp_all
    /-
      🎉 no goals
    -/


/-- If `f` takes a pair of states, but always returns the same value for both elements of the
    pair, then we can simplify to just a single element of state
  -/
@[simp]
theorem mapAccumr₂_redundant_pair (f : α → β → (σ × σ) → (σ × σ) × γ)
    (h : ∀ x y s, let s' := (f x y (s, s)).fst; s'.fst = s'.snd) :
    (mapAccumr₂ f xs ys (s, s)).snd = (mapAccumr₂ (fun x y (s : σ) =>
      (f x y (s, s) |>.fst.fst, f x y (s, s) |>.snd)
    ) xs ys s).snd :=
  mapAccumr₂_bisim_tail <| by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      σ : Type u_5
      n : Nat
      s : σ
      xs : List.Vector α n
      ys : List.Vector β n
      f : α → β → Prod σ σ → Prod (Prod σ σ) γ
      h :
        ∀ (x : α) (y : β) (s : σ),
          let s' := (f x y { fst := s, snd := s }).1;
          Eq s'.1 s'.2
      ⊢ Exists fun R => And (R { fst := s, snd := s } s) (∀ {s : Prod σ σ} {q : σ} ( …
    -/
    use fun (s₁, s₂) s => s₂ = s ∧ s₁ = s
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      σ : Type u_5
      n : Nat
      s : σ
      xs : List.Vector α n
      ys : List.Vector β n
      f : α → β → Prod σ σ → Prod (Prod σ σ) γ
      h :
        ∀ (x : α) (y : β) (s : σ),
          let s' := (f x y { fst := s, snd := s }).1;
          Eq s'.1 s'.2
      ⊢ And ((fun x s => List.Vector.mapAccumr_redundant_pair.match_1 (fun x => Prop …
    -/
    simp_all
    /-
      🎉 no goals
    -/


/--
  If `f` returns the same output and next state for every value of it's first argument, then
  `xs : Vector` is ignored, and we can rewrite `mapAccumr₂` into `map`
-/
@[simp]
theorem mapAccumr₂_unused_input_left [Inhabited α] (f : α → β → σ → σ × γ)
    (h : ∀ a b s, f default b s = f a b s) :
    mapAccumr₂ f xs ys s = mapAccumr (fun b s => f default b s) ys s := by
  induction xs, ys using Vector.revInductionOn₂ generalizing s with
  | nil => rfl
  | snoc xs ys x y ih => simp [h x y s, ih]


/--
  If `f` returns the same output and next state for every value of it's second argument, then
  `ys : Vector` is ignored, and we can rewrite `mapAccumr₂` into `map`
-/
@[simp]
theorem mapAccumr₂_unused_input_right [Inhabited β] (f : α → β → σ → σ × γ)
    (h : ∀ a b s, f a default s = f a b s) :
    mapAccumr₂ f xs ys s = mapAccumr (fun a s => f a default s) xs s := by
  induction xs, ys using Vector.revInductionOn₂ generalizing s with
  | nil => rfl
  | snoc xs ys x y ih => simp [h x y s, ih]


theorem map₂_comm (f : α → α → β) (comm : ∀ a₁ a₂, f a₁ a₂ = f a₂ a₁) :
    map₂ f xs ys = map₂ f ys xs := by
  /-
    α : Type u_1
    β : Type u_2
    n : Nat
    xs ys : List.Vector α n
    f : α → α → β
    comm : ∀ (a₁ a₂ : α), Eq (f a₁ a₂) (f a₂ a₁)
    ⊢ Eq (List.Vector.map₂ f xs ys) (List.Vector.map₂ f ys xs)
  -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  induction xs, ys using Vector.inductionOn₂ <;> simp_all
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem mapAccumr₂_comm (f : α → α → σ → σ × γ) (comm : ∀ a₁ a₂ s, f a₁ a₂ s = f a₂ a₁ s) :
    mapAccumr₂ f xs ys s = mapAccumr₂ f ys xs s := by
  /-
    α : Type u_1
    γ : Type u_3
    σ : Type u_5
    n : Nat
    s : σ
    xs ys : List.Vector α n
    f : α → α → σ → Prod σ γ
    comm : ∀ (a₁ a₂ : α) (s : σ), Eq (f a₁ a₂ s) (f a₂ a₁ s)
    ⊢ Eq (List.Vector.mapAccumr₂ f xs ys s) (List.Vector.mapAccumr₂ f ys xs s)
  -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
  induction xs, ys using Vector.inductionOn₂ generalizing s <;> simp_all
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem map₂_flip (f : α → β → γ) :
    map₂ f xs ys = map₂ (flip f) ys xs := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    n : Nat
    xs : List.Vector α n
    ys : List.Vector β n
    f : α → β → γ
    ⊢ Eq (List.Vector.map₂ f xs ys) (List.Vector.map₂ (flip f) ys xs)
  -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  induction xs, ys using Vector.inductionOn₂ <;> simp_all[flip]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem mapAccumr₂_flip (f : α → β → σ → σ × γ) :
    mapAccumr₂ f xs ys s = mapAccumr₂ (flip f) ys xs s := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ : Type u_5
    n : Nat
    s : σ
    xs : List.Vector α n
    ys : List.Vector β n
    f : α → β → σ → Prod σ γ
    ⊢ Eq (List.Vector.mapAccumr₂ f xs ys s) (List.Vector.mapAccumr₂ (flip f) ys xs …
  -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  induction xs, ys using Vector.inductionOn₂ <;> simp_all[flip]
                                                 /-
                                                   🎉 no goals
                                                 -/


