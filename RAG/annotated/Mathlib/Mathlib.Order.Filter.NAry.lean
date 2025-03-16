/-- The image of a binary function `m : α → β → γ` as a function `Filter α → Filter β → Filter γ`.
Mathematically this should be thought of as the image of the corresponding function `α × β → γ`. -/
def map₂ (m : α → β → γ) (f : Filter α) (g : Filter β) : Filter γ :=
  ((f ×ˢ g).map (uncurry m)).copy { s | ∃ u ∈ f, ∃ v ∈ g, image2 m u v ⊆ s } fun _ ↦ by
    /-
      α : Type u_1
      α' : Type u_2
      β : Type u_3
      β' : Type u_4
      γ : Type u_5
      γ' : Type u_6
      δ : Type u_7
      δ' : Type u_8
      ε : Type u_9
      ε' : Type u_10
      m✝ : α → β → γ
      f✝ f₁ f₂ : Filter α
      g✝ g₁ g₂ : Filter β
      h : Filter γ
      s : Set α
      t : Set β
      u : Set γ
      a : α
      b : β
      m : α → β → γ
      f : Filter α
      g : Filter β
      x✝ : Set γ
      ⊢ Iff (Membership.mem (setOf fun s => Exists fun u => And (Membership.mem f u) …
    -/
    simp only [mem_map, mem_prod_iff, image2_subset_iff, prod_subset_iff]; rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp 900]
theorem mem_map₂_iff : u ∈ map₂ m f g ↔ ∃ s ∈ f, ∃ t ∈ g, image2 m s t ⊆ u :=
  Iff.rfl


theorem image2_mem_map₂ (hs : s ∈ f) (ht : t ∈ g) : image2 m s t ∈ map₂ m f g :=
  ⟨_, hs, _, ht, Subset.rfl⟩


theorem map_prod_eq_map₂ (m : α → β → γ) (f : Filter α) (g : Filter β) :
    Filter.map (fun p : α × β => m p.1 p.2) (f ×ˢ g) = map₂ m f g := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    m : α → β → γ
    f : Filter α
    g : Filter β
    ⊢ Eq (Filter.map (fun p => m p.1 p.2) (SProd.sprod f g)) (Filter.map₂ m f g)
  -/
  rw [map₂, copy_eq, uncurry_def]
  /-
    🎉 no goals
  -/


theorem map_prod_eq_map₂' (m : α × β → γ) (f : Filter α) (g : Filter β) :
    Filter.map m (f ×ˢ g) = map₂ (fun a b => m (a, b)) f g :=
  map_prod_eq_map₂ m.curry f g


@[simp]
theorem map₂_mk_eq_prod (f : Filter α) (g : Filter β) : map₂ Prod.mk f g = f ×ˢ g := by
  /-
    α : Type u_1
    β : Type u_3
    f : Filter α
    g : Filter β
    ⊢ Eq (Filter.map₂ Prod.mk f g) (SProd.sprod f g)
  -/
  simp only [← map_prod_eq_map₂, map_id']
  /-
    🎉 no goals
  -/

-- lemma image2_mem_map₂_iff (hm : injective2 m) : image2 m s t ∈ map₂ m f g ↔ s ∈ f ∧ t ∈ g :=
-- ⟨by { rintro ⟨u, v, hu, hv, h⟩, rw image2_subset_image2_iff hm at h,
--   exact ⟨mem_of_superset hu h.1, mem_of_superset hv h.2⟩ }, fun h ↦ image2_mem_map₂ h.1 h.2⟩

@[gcongr]
theorem map₂_mono (hf : f₁ ≤ f₂) (hg : g₁ ≤ g₂) : map₂ m f₁ g₁ ≤ map₂ m f₂ g₂ :=
  fun _ ⟨s, hs, t, ht, hst⟩ => ⟨s, hf hs, t, hg ht, hst⟩


@[gcongr]
theorem map₂_mono_left (h : g₁ ≤ g₂) : map₂ m f g₁ ≤ map₂ m f g₂ :=
  map₂_mono Subset.rfl h


@[gcongr]
theorem map₂_mono_right (h : f₁ ≤ f₂) : map₂ m f₁ g ≤ map₂ m f₂ g :=
  map₂_mono h Subset.rfl


@[simp]
theorem le_map₂_iff {h : Filter γ} :
    h ≤ map₂ m f g ↔ ∀ ⦃s⦄, s ∈ f → ∀ ⦃t⦄, t ∈ g → image2 m s t ∈ h :=
  ⟨fun H _ hs _ ht => H <| image2_mem_map₂ hs ht, fun H _ ⟨_, hs, _, ht, hu⟩ =>
    mem_of_superset (H hs ht) hu⟩


@[simp]
                                                               /-
                                                                 α : Type u_1
                                                                 β : Type u_3
                                                                 γ : Type u_5
                                                                 m : α → β → γ
                                                                 f : Filter α
                                                                 g : Filter β
                                                                 ⊢ Iff (Eq (Filter.map₂ m f g) Bot.bot) (Or (Eq f Bot.bot) (Eq g Bot.bot))
                                                               -/
theorem map₂_eq_bot_iff : map₂ m f g = ⊥ ↔ f = ⊥ ∨ g = ⊥ := by simp [← map_prod_eq_map₂]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem map₂_bot_left : map₂ m ⊥ g = ⊥ := map₂_eq_bot_iff.2 <| .inl rfl


@[simp]
theorem map₂_bot_right : map₂ m f ⊥ = ⊥ := map₂_eq_bot_iff.2 <| .inr rfl


@[simp]
                                                                      /-
                                                                        α : Type u_1
                                                                        β : Type u_3
                                                                        γ : Type u_5
                                                                        m : α → β → γ
                                                                        f : Filter α
                                                                        g : Filter β
                                                                        ⊢ Iff (Filter.map₂ m f g).NeBot (And f.NeBot g.NeBot)
                                                                      -/
theorem map₂_neBot_iff : (map₂ m f g).NeBot ↔ f.NeBot ∧ g.NeBot := by simp [neBot_iff, not_or]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


protected theorem NeBot.map₂ (hf : f.NeBot) (hg : g.NeBot) : (map₂ m f g).NeBot :=
  map₂_neBot_iff.2 ⟨hf, hg⟩


instance map₂.neBot [NeBot f] [NeBot g] : NeBot (map₂ m f g) := .map₂ ‹_› ‹_›


theorem NeBot.of_map₂_left (h : (map₂ m f g).NeBot) : f.NeBot :=
  (map₂_neBot_iff.1 h).1


theorem NeBot.of_map₂_right (h : (map₂ m f g).NeBot) : g.NeBot :=
  (map₂_neBot_iff.1 h).2


theorem map₂_sup_left : map₂ m (f₁ ⊔ f₂) g = map₂ m f₁ g ⊔ map₂ m f₂ g := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    m : α → β → γ
    f₁ f₂ : Filter α
    g : Filter β
    ⊢ Eq (Filter.map₂ m (Max.max f₁ f₂) g) (Max.max (Filter.map₂ m f₁ g) (Filter.m …
  -/
  simp_rw [← map_prod_eq_map₂, sup_prod, map_sup]
  /-
    🎉 no goals
  -/


theorem map₂_sup_right : map₂ m f (g₁ ⊔ g₂) = map₂ m f g₁ ⊔ map₂ m f g₂ := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    m : α → β → γ
    f : Filter α
    g₁ g₂ : Filter β
    ⊢ Eq (Filter.map₂ m f (Max.max g₁ g₂)) (Max.max (Filter.map₂ m f g₁) (Filter.m …
  -/
  simp_rw [← map_prod_eq_map₂, prod_sup, map_sup]
  /-
    🎉 no goals
  -/


theorem map₂_inf_subset_left : map₂ m (f₁ ⊓ f₂) g ≤ map₂ m f₁ g ⊓ map₂ m f₂ g :=
  Monotone.map_inf_le (fun _ _ ↦ map₂_mono_right) f₁ f₂


theorem map₂_inf_subset_right : map₂ m f (g₁ ⊓ g₂) ≤ map₂ m f g₁ ⊓ map₂ m f g₂ :=
  Monotone.map_inf_le (fun _ _ ↦ map₂_mono_left) g₁ g₂


@[simp]
theorem map₂_pure_left : map₂ m (pure a) g = g.map (m a) := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    m : α → β → γ
    g : Filter β
    a : α
    ⊢ Eq (Filter.map₂ m (Pure.pure a) g) (Filter.map (m a) g)
  -/
  rw [← map_prod_eq_map₂, pure_prod, map_map]; rfl
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem map₂_pure_right : map₂ m f (pure b) = f.map (m · b) := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    m : α → β → γ
    f : Filter α
    b : β
    ⊢ Eq (Filter.map₂ m f (Pure.pure b)) (Filter.map (fun x => m x b) f)
  -/
  rw [← map_prod_eq_map₂, prod_pure, map_map]; rfl
                                               /-
                                                 🎉 no goals
                                               -/


                                                                  /-
                                                                    α : Type u_1
                                                                    β : Type u_3
                                                                    γ : Type u_5
                                                                    m : α → β → γ
                                                                    a : α
                                                                    b : β
                                                                    ⊢ Eq (Filter.map₂ m (Pure.pure a) (Pure.pure b)) (Pure.pure (m a b))
                                                                  -/
theorem map₂_pure : map₂ m (pure a) (pure b) = pure (m a b) := by rw [map₂_pure_right, map_pure]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem map₂_swap (m : α → β → γ) (f : Filter α) (g : Filter β) :
    map₂ m f g = map₂ (fun a b => m b a) g f := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    m : α → β → γ
    f : Filter α
    g : Filter β
    ⊢ Eq (Filter.map₂ m f g) (Filter.map₂ (fun a b => m b a) g f)
  -/
  rw [← map_prod_eq_map₂, prod_comm, map_map, ← map_prod_eq_map₂, Function.comp_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem map₂_left [NeBot g] : map₂ (fun x _ => x) f g = f := by
  /-
    α : Type u_1
    β : Type u_3
    f : Filter α
    g : Filter β
    inst✝ : g.NeBot
    ⊢ Eq (Filter.map₂ (fun x x_1 => x) f g) f
  -/
  rw [← map_prod_eq_map₂, map_fst_prod]
  /-
    🎉 no goals
  -/


@[simp]
                                                                 /-
                                                                   α : Type u_1
                                                                   β : Type u_3
                                                                   f : Filter α
                                                                   g : Filter β
                                                                   inst✝ : f.NeBot
                                                                   ⊢ Eq (Filter.map₂ (fun x y => y) f g) g
                                                                 -/
theorem map₂_right [NeBot f] : map₂ (fun _ y => y) f g = g := by rw [map₂_swap, map₂_left]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem map_map₂ (m : α → β → γ) (n : γ → δ) :
    (map₂ m f g).map n = map₂ (fun a b => n (m a b)) f g := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    f : Filter α
    g : Filter β
    m : α → β → γ
    n : γ → δ
    ⊢ Eq (Filter.map n (Filter.map₂ m f g)) (Filter.map₂ (fun a b => n (m a b)) f g)
  -/
  rw [← map_prod_eq_map₂, ← map_prod_eq_map₂, map_map]; rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem map₂_map_left (m : γ → β → δ) (n : α → γ) :
    map₂ m (f.map n) g = map₂ (fun a b => m (n a) b) f g := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    f : Filter α
    g : Filter β
    m : γ → β → δ
    n : α → γ
    ⊢ Eq (Filter.map₂ m (Filter.map n f) g) (Filter.map₂ (fun a b => m (n a) b) f g)
  -/
  rw [← map_prod_eq_map₂, ← map_prod_eq_map₂, ← @map_id _ g, prod_map_map_eq, map_map, map_id]; rfl
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


theorem map₂_map_right (m : α → γ → δ) (n : β → γ) :
    map₂ m f (g.map n) = map₂ (fun a b => m a (n b)) f g := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    f : Filter α
    g : Filter β
    m : α → γ → δ
    n : β → γ
    ⊢ Eq (Filter.map₂ m f (Filter.map n g)) (Filter.map₂ (fun a b => m a (n b)) f g)
  -/
  rw [map₂_swap, map₂_map_left, map₂_swap]
  /-
    🎉 no goals
  -/


@[simp]
theorem map₂_curry (m : α × β → γ) (f : Filter α) (g : Filter β) :
    map₂ m.curry f g = (f ×ˢ g).map m :=
  (map_prod_eq_map₂' _ _ _).symm


@[simp]
theorem map_uncurry_prod (m : α → β → γ) (f : Filter α) (g : Filter β) :
    (f ×ˢ g).map (uncurry m) = map₂ m f g :=
  (map₂_curry (uncurry m) f g).symm


theorem map₂_assoc {m : δ → γ → ε} {n : α → β → δ} {m' : α → ε' → ε} {n' : β → γ → ε'}
    {h : Filter γ} (h_assoc : ∀ a b c, m (n a b) c = m' a (n' b c)) :
    map₂ m (map₂ n f g) h = map₂ m' f (map₂ n' g h) := by
  rw [← map_prod_eq_map₂ n, ← map_prod_eq_map₂ n', map₂_map_left, map₂_map_right,
    ← map_prod_eq_map₂, ← map_prod_eq_map₂, ← prod_assoc, map_map]
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    ε : Type u_9
    ε' : Type u_10
    f : Filter α
    g : Filter β
    m : δ → γ → ε
    n : α → β → δ
    m' : α → ε' → ε
    n' : β → γ → ε'
    h : Filter γ
    h_assoc : ∀ (a : α) (b : β) (c : γ), Eq (m (n a b) c) (m' a (n' b c))
    ⊢ Eq (Filter.map (fun p => m (n p.1.1 p.1.2) p.2) (SProd.sprod (SProd.sprod f  …
  -/
  simp only [h_assoc, Function.comp_def, Equiv.prodAssoc_apply]
  /-
    🎉 no goals
  -/


theorem map₂_comm {n : β → α → γ} (h_comm : ∀ a b, m a b = n b a) : map₂ m f g = map₂ n g f :=
                                /-
                                  α : Type u_1
                                  β : Type u_3
                                  γ : Type u_5
                                  m : α → β → γ
                                  f : Filter α
                                  g : Filter β
                                  n : β → α → γ
                                  h_comm : ∀ (a : α) (b : β), Eq (m a b) (n b a)
                                  ⊢ Eq (Filter.map₂ (fun a b => m b a) g f) (Filter.map₂ n g f)
                                -/
  (map₂_swap _ _ _).trans <| by simp_rw [h_comm]
                                /-
                                  🎉 no goals
                                -/


theorem map₂_left_comm {m : α → δ → ε} {n : β → γ → δ} {m' : α → γ → δ'} {n' : β → δ' → ε}
    (h_left_comm : ∀ a b c, m a (n b c) = n' b (m' a c)) :
    map₂ m f (map₂ n g h) = map₂ n' g (map₂ m' f h) := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    δ' : Type u_8
    ε : Type u_9
    f : Filter α
    g : Filter β
    h : Filter γ
    m : α → δ → ε
    n : β → γ → δ
    m' : α → γ → δ'
    n' : β → δ' → ε
    h_left_comm : ∀ (a : α) (b : β) (c : γ), Eq (m a (n b c)) (n' b (m' a c))
    ⊢ Eq (Filter.map₂ m f (Filter.map₂ n g h)) (Filter.map₂ n' g (Filter.map₂ m' f …
  -/
  rw [map₂_swap m', map₂_swap m]
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    δ' : Type u_8
    ε : Type u_9
    f : Filter α
    g : Filter β
    h : Filter γ
    m : α → δ → ε
    n : β → γ → δ
    m' : α → γ → δ'
    n' : β → δ' → ε
    h_left_comm : ∀ (a : α) (b : β) (c : γ), Eq (m a (n b c)) (n' b (m' a c))
    ⊢ Eq (Filter.map₂ (fun a b => m b a) (Filter.map₂ n g h) f) (Filter.map₂ n' g  …
  -/
  exact map₂_assoc fun _ _ _ => h_left_comm _ _ _
  /-
    🎉 no goals
  -/


theorem map₂_right_comm {m : δ → γ → ε} {n : α → β → δ} {m' : α → γ → δ'} {n' : δ' → β → ε}
    (h_right_comm : ∀ a b c, m (n a b) c = n' (m' a c) b) :
    map₂ m (map₂ n f g) h = map₂ n' (map₂ m' f h) g := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    δ' : Type u_8
    ε : Type u_9
    f : Filter α
    g : Filter β
    h : Filter γ
    m : δ → γ → ε
    n : α → β → δ
    m' : α → γ → δ'
    n' : δ' → β → ε
    h_right_comm : ∀ (a : α) (b : β) (c : γ), Eq (m (n a b) c) (n' (m' a c) b)
    ⊢ Eq (Filter.map₂ m (Filter.map₂ n f g) h) (Filter.map₂ n' (Filter.map₂ m' f h …
  -/
  rw [map₂_swap n, map₂_swap n']
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    δ' : Type u_8
    ε : Type u_9
    f : Filter α
    g : Filter β
    h : Filter γ
    m : δ → γ → ε
    n : α → β → δ
    m' : α → γ → δ'
    n' : δ' → β → ε
    h_right_comm : ∀ (a : α) (b : β) (c : γ), Eq (m (n a b) c) (n' (m' a c) b)
    ⊢ Eq (Filter.map₂ m (Filter.map₂ (fun a b => n b a) g f) h) (Filter.map₂ (fun  …
  -/
  exact map₂_assoc fun _ _ _ => h_right_comm _ _ _
  /-
    🎉 no goals
  -/


theorem map_map₂_distrib {n : γ → δ} {m' : α' → β' → δ} {n₁ : α → α'} {n₂ : β → β'}
    (h_distrib : ∀ a b, n (m a b) = m' (n₁ a) (n₂ b)) :
    (map₂ m f g).map n = map₂ m' (f.map n₁) (g.map n₂) := by
  /-
    α : Type u_1
    α' : Type u_2
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    δ : Type u_7
    m : α → β → γ
    f : Filter α
    g : Filter β
    n : γ → δ
    m' : α' → β' → δ
    n₁ : α → α'
    n₂ : β → β'
    h_distrib : ∀ (a : α) (b : β), Eq (n (m a b)) (m' (n₁ a) (n₂ b))
    ⊢ Eq (Filter.map n (Filter.map₂ m f g)) (Filter.map₂ m' (Filter.map n₁ f) (Fil …
  -/
  simp_rw [map_map₂, map₂_map_left, map₂_map_right, h_distrib]
  /-
    🎉 no goals
  -/


/-- Symmetric statement to `Filter.map₂_map_left_comm`. -/
theorem map_map₂_distrib_left {n : γ → δ} {m' : α' → β → δ} {n' : α → α'}
    (h_distrib : ∀ a b, n (m a b) = m' (n' a) b) : (map₂ m f g).map n = map₂ m' (f.map n') g :=
  map_map₂_distrib h_distrib


/-- Symmetric statement to `Filter.map_map₂_right_comm`. -/
theorem map_map₂_distrib_right {n : γ → δ} {m' : α → β' → δ} {n' : β → β'}
    (h_distrib : ∀ a b, n (m a b) = m' a (n' b)) : (map₂ m f g).map n = map₂ m' f (g.map n') :=
  map_map₂_distrib h_distrib


/-- Symmetric statement to `Filter.map_map₂_distrib_left`. -/
theorem map₂_map_left_comm {m : α' → β → γ} {n : α → α'} {m' : α → β → δ} {n' : δ → γ}
    (h_left_comm : ∀ a b, m (n a) b = n' (m' a b)) : map₂ m (f.map n) g = (map₂ m' f g).map n' :=
  (map_map₂_distrib_left fun a b => (h_left_comm a b).symm).symm


/-- Symmetric statement to `Filter.map_map₂_distrib_right`. -/
theorem map_map₂_right_comm {m : α → β' → γ} {n : β → β'} {m' : α → β → δ} {n' : δ → γ}
    (h_right_comm : ∀ a b, m a (n b) = n' (m' a b)) : map₂ m f (g.map n) = (map₂ m' f g).map n' :=
  (map_map₂_distrib_right fun a b => (h_right_comm a b).symm).symm


/-- The other direction does not hold because of the `f-f` cross terms on the RHS. -/
theorem map₂_distrib_le_left {m : α → δ → ε} {n : β → γ → δ} {m₁ : α → β → β'} {m₂ : α → γ → γ'}
    {n' : β' → γ' → ε} (h_distrib : ∀ a b c, m a (n b c) = n' (m₁ a b) (m₂ a c)) :
    map₂ m f (map₂ n g h) ≤ map₂ n' (map₂ m₁ f g) (map₂ m₂ f h) := by
  /-
    α : Type u_1
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    γ' : Type u_6
    δ : Type u_7
    ε : Type u_9
    f : Filter α
    g : Filter β
    h : Filter γ
    m : α → δ → ε
    n : β → γ → δ
    m₁ : α → β → β'
    m₂ : α → γ → γ'
    n' : β' → γ' → ε
    h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (m a (n b c)) (n' (m₁ a b) (m₂ a c))
    ⊢ LE.le (Filter.map₂ m f (Filter.map₂ n g h)) (Filter.map₂ n' (Filter.map₂ m₁  …
  -/
  rintro s ⟨t₁, ⟨u₁, hu₁, v, hv, ht₁⟩, t₂, ⟨u₂, hu₂, w, hw, ht₂⟩, hs⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    γ' : Type u_6
    δ : Type u_7
    ε : Type u_9
    f : Filter α
    g : Filter β
    h : Filter γ
    m : α → δ → ε
    n : β → γ → δ
    m₁ : α → β → β'
    m₂ : α → γ → γ'
    n' : β' → γ' → ε
    h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (m a (n b c)) (n' (m₁ a b) (m₂ a c))
    s : Set ε
    t₁ : Set β'
    u₁ : Set α
    hu₁ : Membership.mem f u₁
    v : Set β
    hv : Membership.mem g v
    ht₁ : HasSubset.Subset (Set.image2 m₁ u₁ v) t₁
    t₂ : Set γ'
    hs : HasSubset.Subset (Set.image2 n' t₁ t₂) s
    u₂ : Set α
    hu₂ : Membership.mem f u₂
    w : Set γ
    hw : Membership.mem h w
    ht₂ : HasSubset.Subset (Set.image2 m₂ u₂ w) t₂
    ⊢ Membership.mem (Filter.map₂ m f (Filter.map₂ n g h)) s
  -/
  refine ⟨u₁ ∩ u₂, inter_mem hu₁ hu₂, _, image2_mem_map₂ hv hw, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    γ' : Type u_6
    δ : Type u_7
    ε : Type u_9
    f : Filter α
    g : Filter β
    h : Filter γ
    m : α → δ → ε
    n : β → γ → δ
    m₁ : α → β → β'
    m₂ : α → γ → γ'
    n' : β' → γ' → ε
    h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (m a (n b c)) (n' (m₁ a b) (m₂ a c))
    s : Set ε
    t₁ : Set β'
    u₁ : Set α
    hu₁ : Membership.mem f u₁
    v : Set β
    hv : Membership.mem g v
    ht₁ : HasSubset.Subset (Set.image2 m₁ u₁ v) t₁
    t₂ : Set γ'
    hs : HasSubset.Subset (Set.image2 n' t₁ t₂) s
    u₂ : Set α
    hu₂ : Membership.mem f u₂
    w : Set γ
    hw : Membership.mem h w
    ht₂ : HasSubset.Subset (Set.image2 m₂ u₂ w) t₂
    ⊢ HasSubset.Subset (Set.image2 m (Inter.inter u₁ u₂) (Set.image2 n v w)) s
  -/
  refine (image2_distrib_subset_left h_distrib).trans ((image2_subset ?_ ?_).trans hs)
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
      α : Type u_1
      β : Type u_3
      β' : Type u_4
      γ : Type u_5
      γ' : Type u_6
      δ : Type u_7
      ε : Type u_9
      f : Filter α
      g : Filter β
      h : Filter γ
      m : α → δ → ε
      n : β → γ → δ
      m₁ : α → β → β'
      m₂ : α → γ → γ'
      n' : β' → γ' → ε
      h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (m a (n b c)) (n' (m₁ a b) (m₂ a c))
      s : Set ε
      t₁ : Set β'
      u₁ : Set α
      hu₁ : Membership.mem f u₁
      v : Set β
      hv : Membership.mem g v
      ht₁ : HasSubset.Subset (Set.image2 m₁ u₁ v) t₁
      t₂ : Set γ'
      hs : HasSubset.Subset (Set.image2 n' t₁ t₂) s
      u₂ : Set α
      hu₂ : Membership.mem f u₂
      w : Set γ
      hw : Membership.mem h w
      ht₂ : HasSubset.Subset (Set.image2 m₂ u₂ w) t₂
      ⊢ HasSubset.Subset (Set.image2 m₁ (Inter.inter u₁ u₂) v) t₁
    -/
  · exact (image2_subset_right inter_subset_left).trans ht₁
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
      α : Type u_1
      β : Type u_3
      β' : Type u_4
      γ : Type u_5
      γ' : Type u_6
      δ : Type u_7
      ε : Type u_9
      f : Filter α
      g : Filter β
      h : Filter γ
      m : α → δ → ε
      n : β → γ → δ
      m₁ : α → β → β'
      m₂ : α → γ → γ'
      n' : β' → γ' → ε
      h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (m a (n b c)) (n' (m₁ a b) (m₂ a c))
      s : Set ε
      t₁ : Set β'
      u₁ : Set α
      hu₁ : Membership.mem f u₁
      v : Set β
      hv : Membership.mem g v
      ht₁ : HasSubset.Subset (Set.image2 m₁ u₁ v) t₁
      t₂ : Set γ'
      hs : HasSubset.Subset (Set.image2 n' t₁ t₂) s
      u₂ : Set α
      hu₂ : Membership.mem f u₂
      w : Set γ
      hw : Membership.mem h w
      ht₂ : HasSubset.Subset (Set.image2 m₂ u₂ w) t₂
      ⊢ HasSubset.Subset (Set.image2 m₂ (Inter.inter u₁ u₂) w) t₂
    -/
  · exact (image2_subset_right inter_subset_right).trans ht₂
    /-
      🎉 no goals
    -/


/-- The other direction does not hold because of the `h`-`h` cross terms on the RHS. -/
theorem map₂_distrib_le_right {m : δ → γ → ε} {n : α → β → δ} {m₁ : α → γ → α'} {m₂ : β → γ → β'}
    {n' : α' → β' → ε} (h_distrib : ∀ a b c, m (n a b) c = n' (m₁ a c) (m₂ b c)) :
    map₂ m (map₂ n f g) h ≤ map₂ n' (map₂ m₁ f h) (map₂ m₂ g h) := by
  /-
    α : Type u_1
    α' : Type u_2
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    δ : Type u_7
    ε : Type u_9
    f : Filter α
    g : Filter β
    h : Filter γ
    m : δ → γ → ε
    n : α → β → δ
    m₁ : α → γ → α'
    m₂ : β → γ → β'
    n' : α' → β' → ε
    h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (m (n a b) c) (n' (m₁ a c) (m₂ b c))
    ⊢ LE.le (Filter.map₂ m (Filter.map₂ n f g) h) (Filter.map₂ n' (Filter.map₂ m₁  …
  -/
  rintro s ⟨t₁, ⟨u, hu, w₁, hw₁, ht₁⟩, t₂, ⟨v, hv, w₂, hw₂, ht₂⟩, hs⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    α' : Type u_2
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    δ : Type u_7
    ε : Type u_9
    f : Filter α
    g : Filter β
    h : Filter γ
    m : δ → γ → ε
    n : α → β → δ
    m₁ : α → γ → α'
    m₂ : β → γ → β'
    n' : α' → β' → ε
    h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (m (n a b) c) (n' (m₁ a c) (m₂ b c))
    s : Set ε
    t₁ : Set α'
    u : Set α
    hu : Membership.mem f u
    w₁ : Set γ
    hw₁ : Membership.mem h w₁
    ht₁ : HasSubset.Subset (Set.image2 m₁ u w₁) t₁
    t₂ : Set β'
    hs : HasSubset.Subset (Set.image2 n' t₁ t₂) s
    v : Set β
    hv : Membership.mem g v
    w₂ : Set γ
    hw₂ : Membership.mem h w₂
    ht₂ : HasSubset.Subset (Set.image2 m₂ v w₂) t₂
    ⊢ Membership.mem (Filter.map₂ m (Filter.map₂ n f g) h) s
  -/
  refine ⟨_, image2_mem_map₂ hu hv, w₁ ∩ w₂, inter_mem hw₁ hw₂, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    α' : Type u_2
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    δ : Type u_7
    ε : Type u_9
    f : Filter α
    g : Filter β
    h : Filter γ
    m : δ → γ → ε
    n : α → β → δ
    m₁ : α → γ → α'
    m₂ : β → γ → β'
    n' : α' → β' → ε
    h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (m (n a b) c) (n' (m₁ a c) (m₂ b c))
    s : Set ε
    t₁ : Set α'
    u : Set α
    hu : Membership.mem f u
    w₁ : Set γ
    hw₁ : Membership.mem h w₁
    ht₁ : HasSubset.Subset (Set.image2 m₁ u w₁) t₁
    t₂ : Set β'
    hs : HasSubset.Subset (Set.image2 n' t₁ t₂) s
    v : Set β
    hv : Membership.mem g v
    w₂ : Set γ
    hw₂ : Membership.mem h w₂
    ht₂ : HasSubset.Subset (Set.image2 m₂ v w₂) t₂
    ⊢ HasSubset.Subset (Set.image2 m (Set.image2 n u v) (Inter.inter w₁ w₂)) s
  -/
  refine (image2_distrib_subset_right h_distrib).trans ((image2_subset ?_ ?_).trans hs)
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
      α : Type u_1
      α' : Type u_2
      β : Type u_3
      β' : Type u_4
      γ : Type u_5
      δ : Type u_7
      ε : Type u_9
      f : Filter α
      g : Filter β
      h : Filter γ
      m : δ → γ → ε
      n : α → β → δ
      m₁ : α → γ → α'
      m₂ : β → γ → β'
      n' : α' → β' → ε
      h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (m (n a b) c) (n' (m₁ a c) (m₂ b c))
      s : Set ε
      t₁ : Set α'
      u : Set α
      hu : Membership.mem f u
      w₁ : Set γ
      hw₁ : Membership.mem h w₁
      ht₁ : HasSubset.Subset (Set.image2 m₁ u w₁) t₁
      t₂ : Set β'
      hs : HasSubset.Subset (Set.image2 n' t₁ t₂) s
      v : Set β
      hv : Membership.mem g v
      w₂ : Set γ
      hw₂ : Membership.mem h w₂
      ht₂ : HasSubset.Subset (Set.image2 m₂ v w₂) t₂
      ⊢ HasSubset.Subset (Set.image2 m₁ u (Inter.inter w₁ w₂)) t₁
    -/
  · exact (image2_subset_left inter_subset_left).trans ht₁
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
      α : Type u_1
      α' : Type u_2
      β : Type u_3
      β' : Type u_4
      γ : Type u_5
      δ : Type u_7
      ε : Type u_9
      f : Filter α
      g : Filter β
      h : Filter γ
      m : δ → γ → ε
      n : α → β → δ
      m₁ : α → γ → α'
      m₂ : β → γ → β'
      n' : α' → β' → ε
      h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (m (n a b) c) (n' (m₁ a c) (m₂ b c))
      s : Set ε
      t₁ : Set α'
      u : Set α
      hu : Membership.mem f u
      w₁ : Set γ
      hw₁ : Membership.mem h w₁
      ht₁ : HasSubset.Subset (Set.image2 m₁ u w₁) t₁
      t₂ : Set β'
      hs : HasSubset.Subset (Set.image2 n' t₁ t₂) s
      v : Set β
      hv : Membership.mem g v
      w₂ : Set γ
      hw₂ : Membership.mem h w₂
      ht₂ : HasSubset.Subset (Set.image2 m₂ v w₂) t₂
      ⊢ HasSubset.Subset (Set.image2 m₂ v (Inter.inter w₁ w₂)) t₂
    -/
  · exact (image2_subset_left inter_subset_right).trans ht₂
    /-
      🎉 no goals
    -/


theorem map_map₂_antidistrib {n : γ → δ} {m' : β' → α' → δ} {n₁ : β → β'} {n₂ : α → α'}
    (h_antidistrib : ∀ a b, n (m a b) = m' (n₁ b) (n₂ a)) :
    (map₂ m f g).map n = map₂ m' (g.map n₁) (f.map n₂) := by
  /-
    α : Type u_1
    α' : Type u_2
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    δ : Type u_7
    m : α → β → γ
    f : Filter α
    g : Filter β
    n : γ → δ
    m' : β' → α' → δ
    n₁ : β → β'
    n₂ : α → α'
    h_antidistrib : ∀ (a : α) (b : β), Eq (n (m a b)) (m' (n₁ b) (n₂ a))
    ⊢ Eq (Filter.map n (Filter.map₂ m f g)) (Filter.map₂ m' (Filter.map n₁ g) (Fil …
  -/
  rw [map₂_swap m]
  /-
    α : Type u_1
    α' : Type u_2
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    δ : Type u_7
    m : α → β → γ
    f : Filter α
    g : Filter β
    n : γ → δ
    m' : β' → α' → δ
    n₁ : β → β'
    n₂ : α → α'
    h_antidistrib : ∀ (a : α) (b : β), Eq (n (m a b)) (m' (n₁ b) (n₂ a))
    ⊢ Eq (Filter.map n (Filter.map₂ (fun a b => m b a) g f)) (Filter.map₂ m' (Filt …
  -/
  exact map_map₂_distrib fun _ _ => h_antidistrib _ _
  /-
    🎉 no goals
  -/


/-- Symmetric statement to `Filter.map₂_map_left_anticomm`. -/
theorem map_map₂_antidistrib_left {n : γ → δ} {m' : β' → α → δ} {n' : β → β'}
    (h_antidistrib : ∀ a b, n (m a b) = m' (n' b) a) : (map₂ m f g).map n = map₂ m' (g.map n') f :=
  map_map₂_antidistrib h_antidistrib


/-- Symmetric statement to `Filter.map_map₂_right_anticomm`. -/
theorem map_map₂_antidistrib_right {n : γ → δ} {m' : β → α' → δ} {n' : α → α'}
    (h_antidistrib : ∀ a b, n (m a b) = m' b (n' a)) : (map₂ m f g).map n = map₂ m' g (f.map n') :=
  map_map₂_antidistrib h_antidistrib


/-- Symmetric statement to `Filter.map_map₂_antidistrib_left`. -/
theorem map₂_map_left_anticomm {m : α' → β → γ} {n : α → α'} {m' : β → α → δ} {n' : δ → γ}
    (h_left_anticomm : ∀ a b, m (n a) b = n' (m' b a)) :
    map₂ m (f.map n) g = (map₂ m' g f).map n' :=
  (map_map₂_antidistrib_left fun a b => (h_left_anticomm b a).symm).symm


/-- Symmetric statement to `Filter.map_map₂_antidistrib_right`. -/
theorem map_map₂_right_anticomm {m : α → β' → γ} {n : β → β'} {m' : β → α → δ} {n' : δ → γ}
    (h_right_anticomm : ∀ a b, m a (n b) = n' (m' b a)) :
    map₂ m f (g.map n) = (map₂ m' g f).map n' :=
  (map_map₂_antidistrib_right fun a b => (h_right_anticomm b a).symm).symm


/-- If `a` is a left identity for `f : α → β → β`, then `pure a` is a left identity for
`Filter.map₂ f`. -/
theorem map₂_left_identity {f : α → β → β} {a : α} (h : ∀ b, f a b = b) (l : Filter β) :
                                /-
                                  α : Type u_1
                                  β : Type u_3
                                  f : α → β → β
                                  a : α
                                  h : ∀ (b : β), Eq (f a b) b
                                  l : Filter β
                                  ⊢ Eq (Filter.map₂ f (Pure.pure a) l) l
                                -/
    map₂ f (pure a) l = l := by rw [map₂_pure_left, show f a = id from funext h, map_id]
                                /-
                                  🎉 no goals
                                -/


/-- If `b` is a right identity for `f : α → β → α`, then `pure b` is a right identity for
`Filter.map₂ f`. -/
theorem map₂_right_identity {f : α → β → α} {b : β} (h : ∀ a, f a b = a) (l : Filter α) :
                                /-
                                  α : Type u_1
                                  β : Type u_3
                                  f : α → β → α
                                  b : β
                                  h : ∀ (a : α), Eq (f a b) a
                                  l : Filter α
                                  ⊢ Eq (Filter.map₂ f l (Pure.pure b)) l
                                -/
    map₂ f l (pure b) = l := by rw [map₂_pure_right, funext h, map_id']
                                /-
                                  🎉 no goals
                                -/


