/-- An additive `n`-Freiman homomorphism from a set `A` to a set `B` is a map which preserves sums
of `n` elements. -/
structure IsAddFreimanHom [AddCommMonoid α] [AddCommMonoid β] (n : ℕ) (A : Set α) (B : Set β)
    (f : α → β) : Prop where
  mapsTo : MapsTo f A B
  /-- An additive `n`-Freiman homomorphism preserves sums of `n` elements. -/
  map_sum_eq_map_sum ⦃s t : Multiset α⦄ (hsA : ∀ ⦃x⦄, x ∈ s → x ∈ A) (htA : ∀ ⦃x⦄, x ∈ t → x ∈ A)
    (hs : Multiset.card s = n) (ht : Multiset.card t = n) (h : s.sum = t.sum) :
    (s.map f).sum = (t.map f).sum


/-- An `n`-Freiman homomorphism from a set `A` to a set `B` is a map which preserves products of `n`
elements. -/
@[to_additive]
structure IsMulFreimanHom (n : ℕ) (A : Set α) (B : Set β) (f : α → β) : Prop where
  mapsTo : MapsTo f A B
  /-- An `n`-Freiman homomorphism preserves products of `n` elements. -/
  map_prod_eq_map_prod ⦃s t : Multiset α⦄ (hsA : ∀ ⦃x⦄, x ∈ s → x ∈ A) (htA : ∀ ⦃x⦄, x ∈ t → x ∈ A)
    (hs : Multiset.card s = n) (ht : Multiset.card t = n) (h : s.prod = t.prod) :
    (s.map f).prod = (t.map f).prod


/-- An additive `n`-Freiman homomorphism from a set `A` to a set `B` is a bijective map which
preserves sums of `n` elements. -/
structure IsAddFreimanIso [AddCommMonoid α] [AddCommMonoid β] (n : ℕ) (A : Set α) (B : Set β)
    (f : α → β) : Prop where
  bijOn : BijOn f A B
  /-- An additive `n`-Freiman homomorphism preserves sums of `n` elements. -/
  map_sum_eq_map_sum ⦃s t : Multiset α⦄ (hsA : ∀ ⦃x⦄, x ∈ s → x ∈ A) (htA : ∀ ⦃x⦄, x ∈ t → x ∈ A)
    (hs : Multiset.card s = n) (ht : Multiset.card t = n) :
    (s.map f).sum = (t.map f).sum ↔ s.sum = t.sum


/-- An `n`-Freiman homomorphism from a set `A` to a set `B` is a map which preserves products of `n`
elements. -/
@[to_additive]
structure IsMulFreimanIso (n : ℕ) (A : Set α) (B : Set β) (f : α → β) : Prop where
  bijOn : BijOn f A B
  /-- An `n`-Freiman homomorphism preserves products of `n` elements. -/
  map_prod_eq_map_prod ⦃s t : Multiset α⦄ (hsA : ∀ ⦃x⦄, x ∈ s → x ∈ A) (htA : ∀ ⦃x⦄, x ∈ t → x ∈ A)
    (hs : Multiset.card s = n) (ht : Multiset.card t = n) :
    (s.map f).prod = (t.map f).prod ↔ s.prod = t.prod


@[to_additive]
lemma IsMulFreimanIso.isMulFreimanHom (hf : IsMulFreimanIso n A B f) : IsMulFreimanHom n A B f where
  mapsTo := hf.bijOn.mapsTo
  map_prod_eq_map_prod _s _t hsA htA hs ht := (hf.map_prod_eq_map_prod hsA htA hs ht).2


lemma IsMulFreimanHom.congr (hf₁ : IsMulFreimanHom n A B f₁) (h : EqOn f₁ f₂ A) :
    IsMulFreimanHom n A B f₂ where
  mapsTo := hf₁.mapsTo.congr h
  map_prod_eq_map_prod s t hsA htA hs ht h' := by
    rw [map_congr rfl fun x hx => (h (hsA hx)).symm, map_congr rfl fun x hx => (h (htA hx)).symm,
      hf₁.map_prod_eq_map_prod hsA htA hs ht h']


lemma IsMulFreimanIso.congr (hf₁ : IsMulFreimanIso n A B f₁) (h : EqOn f₁ f₂ A) :
    IsMulFreimanIso n A B f₂ where
  bijOn := hf₁.bijOn.congr h
  map_prod_eq_map_prod s t hsA htA hs ht := by
    rw [map_congr rfl fun x hx => h.symm (hsA hx), map_congr rfl fun x hx => h.symm (htA hx),
      hf₁.map_prod_eq_map_prod hsA htA hs ht]


@[to_additive]
lemma IsMulFreimanHom.mul_eq_mul (hf : IsMulFreimanHom 2 A B f) {a b c d : α}
    (ha : a ∈ A) (hb : b ∈ A) (hc : c ∈ A) (hd : d ∈ A) (h : a * b = c * d) :
    f a * f b = f c * f d := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : CommMonoid β
    A : Set α
    B : Set β
    f : α → β
    hf : IsMulFreimanHom 2 A B f
    a b c d : α
    ha : Membership.mem A a
    hb : Membership.mem A b
    hc : Membership.mem A c
    hd : Membership.mem A d
    h : Eq (HMul.hMul a b) (HMul.hMul c d)
    ⊢ Eq (HMul.hMul (f a) (f b)) (HMul.hMul (f c) (f d))
  -/
  simp_rw [← prod_pair] at h ⊢
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : CommMonoid β
    A : Set α
    B : Set β
    f : α → β
    hf : IsMulFreimanHom 2 A B f
    a b c d : α
    ha : Membership.mem A a
    hb : Membership.mem A b
    hc : Membership.mem A c
    hd : Membership.mem A d
    h : Eq (Insert.insert a (Singleton.singleton b)).prod (Insert.insert c (Single …
    ⊢ Eq (Insert.insert (f a) (Singleton.singleton (f b))).prod (Insert.insert (f  …
  -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  refine hf.map_prod_eq_map_prod ?_ ?_ (card_pair _ _) (card_pair _ _) h <;> simp [ha, hb, hc, hd]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[to_additive]
lemma IsMulFreimanIso.mul_eq_mul (hf : IsMulFreimanIso 2 A B f) {a b c d : α}
    (ha : a ∈ A) (hb : b ∈ A) (hc : c ∈ A) (hd : d ∈ A) :
    f a * f b = f c * f d ↔ a * b = c * d := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : CommMonoid β
    A : Set α
    B : Set β
    f : α → β
    hf : IsMulFreimanIso 2 A B f
    a b c d : α
    ha : Membership.mem A a
    hb : Membership.mem A b
    hc : Membership.mem A c
    hd : Membership.mem A d
    ⊢ Iff (Eq (HMul.hMul (f a) (f b)) (HMul.hMul (f c) (f d))) (Eq (HMul.hMul a b) …
  -/
  simp_rw [← prod_pair]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : CommMonoid β
    A : Set α
    B : Set β
    f : α → β
    hf : IsMulFreimanIso 2 A B f
    a b c d : α
    ha : Membership.mem A a
    hb : Membership.mem A b
    hc : Membership.mem A c
    hd : Membership.mem A d
    ⊢ Iff (Eq (Insert.insert (f a) (Singleton.singleton (f b))).prod (Insert.inser …
  -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  refine hf.map_prod_eq_map_prod ?_ ?_ (card_pair _ _) (card_pair _ _) <;> simp [ha, hb, hc, hd]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- Characterisation of `2`-Freiman homomorphisms. -/
@[to_additive "Characterisation of `2`-Freiman homomorphisms."]
lemma isMulFreimanHom_two :
    IsMulFreimanHom 2 A B f ↔ MapsTo f A B ∧ ∀ a ∈ A, ∀ b ∈ A, ∀ c ∈ A, ∀ d ∈ A,
      a * b = c * d → f a * f b = f c * f d where
  mp hf := ⟨hf.mapsTo, fun _ ha _ hb _ hc _ hd ↦ hf.mul_eq_mul ha hb hc hd⟩
                      /-
                        α : Type u_2
                        β : Type u_3
                        inst✝¹ : CommMonoid α
                        inst✝ : CommMonoid β
                        A : Set α
                        B : Set β
                        f : α → β
                        hf : And (Set.MapsTo f A B) (∀ (a : α), Membership.mem A a → ∀ (b : α), Member …
                        ⊢ ∀ ⦃s t : Multiset α⦄, (∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x) → …
                      -/
  mpr hf := ⟨hf.1, by aesop (add simp card_eq_two)⟩
                      /-
                        🎉 no goals
                      -/


/-- Characterisation of `2`-Freiman homs. -/
@[to_additive "Characterisation of `2`-Freiman isomorphisms."]
lemma isMulFreimanIso_two :
    IsMulFreimanIso 2 A B f ↔ BijOn f A B ∧ ∀ a ∈ A, ∀ b ∈ A, ∀ c ∈ A, ∀ d ∈ A,
      f a * f b = f c * f d ↔ a * b = c * d where
  mp hf := ⟨hf.bijOn, fun _ ha _ hb _ hc _ hd => hf.mul_eq_mul ha hb hc hd⟩
                      /-
                        α : Type u_2
                        β : Type u_3
                        inst✝¹ : CommMonoid α
                        inst✝ : CommMonoid β
                        A : Set α
                        B : Set β
                        f : α → β
                        hf : And (Set.BijOn f A B) (∀ (a : α), Membership.mem A a → ∀ (b : α), Members …
                        ⊢ ∀ ⦃s t : Multiset α⦄, (∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x) → …
                      -/
  mpr hf := ⟨hf.1, by aesop (add simp card_eq_two)⟩
                      /-
                        🎉 no goals
                      -/


@[to_additive] lemma isMulFreimanHom_id (hA : A₁ ⊆ A₂) : IsMulFreimanHom n A₁ A₂ id where
  mapsTo := hA
                                           /-
                                             α : Type u_2
                                             inst✝ : CommMonoid α
                                             A₁ A₂ : Set α
                                             n : Nat
                                             hA : HasSubset.Subset A₁ A₂
                                             s t : Multiset α
                                             x✝³ : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A₁ x
                                             x✝² : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A₁ x
                                             x✝¹ : Eq s.card n
                                             x✝ : Eq t.card n
                                             h : Eq s.prod t.prod
                                             ⊢ Eq (Multiset.map id s).prod (Multiset.map id t).prod
                                           -/
  map_prod_eq_map_prod s t _ _ _ _ h := by simpa using h
                                           /-
                                             🎉 no goals
                                           -/


@[to_additive] lemma isMulFreimanIso_id : IsMulFreimanIso n A A id where
  bijOn := bijOn_id _
                                         /-
                                           α : Type u_2
                                           inst✝ : CommMonoid α
                                           A : Set α
                                           n : Nat
                                           s t : Multiset α
                                           x✝³ : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
                                           x✝² : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
                                           x✝¹ : Eq s.card n
                                           x✝ : Eq t.card n
                                           ⊢ Iff (Eq (Multiset.map id s).prod (Multiset.map id t).prod) (Eq s.prod t.prod)
                                         -/
  map_prod_eq_map_prod s t _ _ _ _ := by simp
                                         /-
                                           🎉 no goals
                                         -/


@[to_additive] lemma IsMulFreimanHom.comp (hg : IsMulFreimanHom n B C g)
    (hf : IsMulFreimanHom n A B f) : IsMulFreimanHom n A C (g ∘ f) where
  mapsTo := hg.mapsTo.comp hf.mapsTo
  map_prod_eq_map_prod s t hsA htA hs ht h := by
    /-
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝² : CommMonoid α
      inst✝¹ : CommMonoid β
      inst✝ : CommMonoid γ
      A : Set α
      B : Set β
      C : Set γ
      f : α → β
      g : β → γ
      n : Nat
      hg : IsMulFreimanHom n B C g
      hf : IsMulFreimanHom n A B f
      s t : Multiset α
      hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
      htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
      hs : Eq s.card n
      ht : Eq t.card n
      h : Eq s.prod t.prod
      ⊢ Eq (Multiset.map (Function.comp g f) s).prod (Multiset.map (Function.comp g  …
    -/
    rw [← map_map, ← map_map]
    refine hg.map_prod_eq_map_prod ?_ ?_ (by rwa [card_map]) (by rwa [card_map])
      (hf.map_prod_eq_map_prod hsA htA hs ht h)
      /-
        case refine_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝² : CommMonoid α
        inst✝¹ : CommMonoid β
        inst✝ : CommMonoid γ
        A : Set α
        B : Set β
        C : Set γ
        f : α → β
        g : β → γ
        n : Nat
        hg : IsMulFreimanHom n B C g
        hf : IsMulFreimanHom n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card n
        ht : Eq t.card n
        h : Eq s.prod t.prod
        ⊢ ∀ ⦃x : β⦄, Membership.mem (Multiset.map f s) x → Membership.mem B x
      -/
    · simpa using fun a h ↦ hf.mapsTo (hsA h)
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝² : CommMonoid α
        inst✝¹ : CommMonoid β
        inst✝ : CommMonoid γ
        A : Set α
        B : Set β
        C : Set γ
        f : α → β
        g : β → γ
        n : Nat
        hg : IsMulFreimanHom n B C g
        hf : IsMulFreimanHom n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card n
        ht : Eq t.card n
        h : Eq s.prod t.prod
        ⊢ ∀ ⦃x : β⦄, Membership.mem (Multiset.map f t) x → Membership.mem B x
      -/
    · simpa using fun a h ↦ hf.mapsTo (htA h)
      /-
        🎉 no goals
      -/


@[to_additive] lemma IsMulFreimanIso.comp (hg : IsMulFreimanIso n B C g)
    (hf : IsMulFreimanIso n A B f) : IsMulFreimanIso n A C (g ∘ f) where
  bijOn := hg.bijOn.comp hf.bijOn
  map_prod_eq_map_prod s t hsA htA hs ht := by
    /-
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝² : CommMonoid α
      inst✝¹ : CommMonoid β
      inst✝ : CommMonoid γ
      A : Set α
      B : Set β
      C : Set γ
      f : α → β
      g : β → γ
      n : Nat
      hg : IsMulFreimanIso n B C g
      hf : IsMulFreimanIso n A B f
      s t : Multiset α
      hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
      htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
      hs : Eq s.card n
      ht : Eq t.card n
      ⊢ Iff (Eq (Multiset.map (Function.comp g f) s).prod (Multiset.map (Function.co …
    -/
    rw [← map_map, ← map_map]
    rw [hg.map_prod_eq_map_prod _ _ (by rwa [card_map]) (by rwa [card_map]),
      hf.map_prod_eq_map_prod hsA htA hs ht]
      /-
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝² : CommMonoid α
        inst✝¹ : CommMonoid β
        inst✝ : CommMonoid γ
        A : Set α
        B : Set β
        C : Set γ
        f : α → β
        g : β → γ
        n : Nat
        hg : IsMulFreimanIso n B C g
        hf : IsMulFreimanIso n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card n
        ht : Eq t.card n
        ⊢ ∀ ⦃x : β⦄, Membership.mem (Multiset.map f s) x → Membership.mem B x
      -/
    · simpa using fun a h ↦ hf.bijOn.mapsTo (hsA h)
      /-
        🎉 no goals
      -/
      /-
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝² : CommMonoid α
        inst✝¹ : CommMonoid β
        inst✝ : CommMonoid γ
        A : Set α
        B : Set β
        C : Set γ
        f : α → β
        g : β → γ
        n : Nat
        hg : IsMulFreimanIso n B C g
        hf : IsMulFreimanIso n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card n
        ht : Eq t.card n
        ⊢ ∀ ⦃x : β⦄, Membership.mem (Multiset.map f t) x → Membership.mem B x
      -/
    · simpa using fun a h ↦ hf.bijOn.mapsTo (htA h)
      /-
        🎉 no goals
      -/


@[to_additive] lemma IsMulFreimanHom.subset (hA : A₁ ⊆ A₂) (hf : IsMulFreimanHom n A₂ B₂ f)
    (hf' : MapsTo f A₁ B₁) : IsMulFreimanHom n A₁ B₁ f where
  mapsTo := hf'
  __ := hf.comp (isMulFreimanHom_id hA)


@[to_additive] lemma IsMulFreimanHom.superset (hB : B₁ ⊆ B₂) (hf : IsMulFreimanHom n A B₁ f) :
    IsMulFreimanHom n A B₂ f := (isMulFreimanHom_id hB).comp hf


@[to_additive] lemma IsMulFreimanIso.subset (hA : A₁ ⊆ A₂) (hf : IsMulFreimanIso n A₂ B₂ f)
    (hf' : BijOn f A₁ B₁) : IsMulFreimanIso n A₁ B₁ f where
  bijOn := hf'
  map_prod_eq_map_prod s t hsA htA hs ht := by
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : CommMonoid α
      inst✝ : CommMonoid β
      A₁ A₂ : Set α
      B₁ B₂ : Set β
      f : α → β
      n : Nat
      hA : HasSubset.Subset A₁ A₂
      hf : IsMulFreimanIso n A₂ B₂ f
      hf' : Set.BijOn f A₁ B₁
      s t : Multiset α
      hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A₁ x
      htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A₁ x
      hs : Eq s.card n
      ht : Eq t.card n
      ⊢ Iff (Eq (Multiset.map f s).prod (Multiset.map f t).prod) (Eq s.prod t.prod)
    -/
    refine hf.map_prod_eq_map_prod (fun a ha ↦ hA (hsA ha)) (fun a ha ↦ hA (htA ha)) hs ht
    /-
      🎉 no goals
    -/


@[to_additive]
lemma isMulFreimanHom_const {b : β} (hb : b ∈ B) : IsMulFreimanHom n A B fun _ ↦ b where
  mapsTo _ _ := hb
                                             /-
                                               α : Type u_2
                                               β : Type u_3
                                               inst✝¹ : CommMonoid α
                                               inst✝ : CommMonoid β
                                               A : Set α
                                               B : Set β
                                               n : Nat
                                               b : β
                                               hb : Membership.mem B b
                                               s t : Multiset α
                                               x✝² : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
                                               x✝¹ : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
                                               hs : Eq s.card n
                                               ht : Eq t.card n
                                               x✝ : Eq s.prod t.prod
                                               ⊢ Eq (Multiset.map (fun x => b) s).prod (Multiset.map (fun x => b) t).prod
                                             -/
  map_prod_eq_map_prod s t _ _ hs ht _ := by simp only [map_const', hs, prod_replicate, ht]
                                             /-
                                               🎉 no goals
                                             -/


@[to_additive (attr := simp)]
lemma isMulFreimanHom_zero_iff : IsMulFreimanHom 0 A B f ↔ MapsTo f A B :=
                                      /-
                                        α : Type u_2
                                        β : Type u_3
                                        inst✝¹ : CommMonoid α
                                        inst✝ : CommMonoid β
                                        A : Set α
                                        B : Set β
                                        f : α → β
                                        h : Set.MapsTo f A B
                                        ⊢ ∀ ⦃s t : Multiset α⦄, (∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x) → …
                                      -/
  ⟨fun h => h.mapsTo, fun h => ⟨h, by aesop⟩⟩
                                      /-
                                        🎉 no goals
                                      -/


@[to_additive (attr := simp)]
lemma isMulFreimanIso_zero_iff : IsMulFreimanIso 0 A B f ↔ BijOn f A B :=
                                     /-
                                       α : Type u_2
                                       β : Type u_3
                                       inst✝¹ : CommMonoid α
                                       inst✝ : CommMonoid β
                                       A : Set α
                                       B : Set β
                                       f : α → β
                                       h : Set.BijOn f A B
                                       ⊢ ∀ ⦃s t : Multiset α⦄, (∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x) → …
                                     -/
  ⟨fun h => h.bijOn, fun h => ⟨h, by aesop⟩⟩
                                     /-
                                       🎉 no goals
                                     -/


@[to_additive (attr := simp) isAddFreimanHom_one_iff]
lemma isMulFreimanHom_one_iff : IsMulFreimanHom 1 A B f ↔ MapsTo f A B :=
                                      /-
                                        α : Type u_2
                                        β : Type u_3
                                        inst✝¹ : CommMonoid α
                                        inst✝ : CommMonoid β
                                        A : Set α
                                        B : Set β
                                        f : α → β
                                        h : Set.MapsTo f A B
                                        ⊢ ∀ ⦃s t : Multiset α⦄, (∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x) → …
                                      -/
  ⟨fun h => h.mapsTo, fun h => ⟨h, by aesop (add simp card_eq_one)⟩⟩
                                      /-
                                        🎉 no goals
                                      -/


@[to_additive (attr := simp) isAddFreimanIso_one_iff]
lemma isMulFreimanIso_one_iff : IsMulFreimanIso 1 A B f ↔ BijOn f A B :=
                                     /-
                                       α : Type u_2
                                       β : Type u_3
                                       inst✝¹ : CommMonoid α
                                       inst✝ : CommMonoid β
                                       A : Set α
                                       B : Set β
                                       f : α → β
                                       h : Set.BijOn f A B
                                       ⊢ ∀ ⦃s t : Multiset α⦄, (∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x) → …
                                     -/
  ⟨fun h => h.bijOn, fun h => ⟨h, by aesop (add simp [card_eq_one, BijOn])⟩⟩
                                     /-
                                       🎉 no goals
                                     -/


@[to_additive (attr := simp)]
lemma isMulFreimanHom_empty : IsMulFreimanHom n (∅ : Set α) B f where
  mapsTo := mapsTo_empty f B
                                 /-
                                   α : Type u_2
                                   β : Type u_3
                                   inst✝¹ : CommMonoid α
                                   inst✝ : CommMonoid β
                                   B : Set β
                                   f : α → β
                                   n : Nat
                                   s t : Multiset α
                                   ⊢ (∀ ⦃x : α⦄, Membership.mem s x → Membership.mem EmptyCollection.emptyCollect …
                                 -/
  map_prod_eq_map_prod s t := by aesop (add simp eq_zero_of_forall_not_mem)
                                 /-
                                   🎉 no goals
                                 -/


@[to_additive (attr := simp)]
lemma isMulFreimanIso_empty : IsMulFreimanIso n (∅ : Set α) (∅ : Set β) f where
  bijOn := bijOn_empty _
  map_prod_eq_map_prod s t hs ht := by
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : CommMonoid α
      inst✝ : CommMonoid β
      f : α → β
      n : Nat
      s t : Multiset α
      hs : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem EmptyCollection.emptyColle …
      ht : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem EmptyCollection.emptyColle …
      ⊢ Eq s.card n → Eq t.card n → Iff (Eq (Multiset.map f s).prod (Multiset.map f  …
    -/
    simp [eq_zero_of_forall_not_mem hs, eq_zero_of_forall_not_mem ht]
    /-
      🎉 no goals
    -/


@[to_additive] lemma IsMulFreimanHom.mul (h₁ : IsMulFreimanHom n A B₁ f₁)
    (h₂ : IsMulFreimanHom n A B₂ f₂) : IsMulFreimanHom n A (B₁ * B₂) (f₁ * f₂) where
  mapsTo := h₁.mapsTo.mul h₂.mapsTo
  map_prod_eq_map_prod s t hsA htA hs ht h := by
    rw [Pi.mul_def, prod_map_mul, prod_map_mul, h₁.map_prod_eq_map_prod hsA htA hs ht h,
      h₂.map_prod_eq_map_prod hsA htA hs ht h]


@[to_additive] lemma MonoidHomClass.isMulFreimanHom [FunLike F α β] [MonoidHomClass F α β] (f : F)
    (hfAB : MapsTo f A B) : IsMulFreimanHom n A B f where
  mapsTo := hfAB
                                           /-
                                             F : Type u_1
                                             α : Type u_2
                                             β : Type u_3
                                             inst✝³ : CommMonoid α
                                             inst✝² : CommMonoid β
                                             A : Set α
                                             B : Set β
                                             n : Nat
                                             inst✝¹ : FunLike F α β
                                             inst✝ : MonoidHomClass F α β
                                             f : F
                                             hfAB : Set.MapsTo (⇑f) A B
                                             s t : Multiset α
                                             x✝³ : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
                                             x✝² : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
                                             x✝¹ : Eq s.card n
                                             x✝ : Eq t.card n
                                             h : Eq s.prod t.prod
                                             ⊢ Eq (Multiset.map (⇑f) s).prod (Multiset.map (⇑f) t).prod
                                           -/
  map_prod_eq_map_prod s t _ _ _ _ h := by rw [← map_multiset_prod, h, map_multiset_prod]
                                           /-
                                             🎉 no goals
                                           -/


@[to_additive] lemma MulEquivClass.isMulFreimanIso [EquivLike F α β] [MulEquivClass F α β] (f : F)
    (hfAB : BijOn f A B) : IsMulFreimanIso n A B f where
  bijOn := hfAB
  map_prod_eq_map_prod s t _ _ _ _ := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝³ : CommMonoid α
      inst✝² : CommMonoid β
      A : Set α
      B : Set β
      n : Nat
      inst✝¹ : EquivLike F α β
      inst✝ : MulEquivClass F α β
      f : F
      hfAB : Set.BijOn (⇑f) A B
      s t : Multiset α
      x✝³ : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
      x✝² : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
      x✝¹ : Eq s.card n
      x✝ : Eq t.card n
      ⊢ Iff (Eq (Multiset.map (⇑f) s).prod (Multiset.map (⇑f) t).prod) (Eq s.prod t. …
    -/
    rw [← map_multiset_prod, ← map_multiset_prod, EquivLike.apply_eq_iff_eq]
    /-
      🎉 no goals
    -/


@[to_additive]
lemma IsMulFreimanHom.subtypeVal {S : Type*} [SetLike S α] [SubmonoidClass S α] {s : S} :
    IsMulFreimanHom n (univ : Set s) univ Subtype.val :=
  MonoidHomClass.isMulFreimanHom (SubmonoidClass.subtype s) (mapsTo_univ ..)


@[to_additive]
lemma IsMulFreimanHom.mono (hmn : m ≤ n) (hf : IsMulFreimanHom n A B f) :
    IsMulFreimanHom m A B f where
  mapsTo := hf.mapsTo
  map_prod_eq_map_prod s t hsA htA hs ht h := by
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : CommMonoid α
      inst✝ : CancelCommMonoid β
      A : Set α
      B : Set β
      f : α → β
      m n : Nat
      hmn : LE.le m n
      hf : IsMulFreimanHom n A B f
      s t : Multiset α
      hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
      htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
      hs : Eq s.card m
      ht : Eq t.card m
      h : Eq s.prod t.prod
      ⊢ Eq (Multiset.map f s).prod (Multiset.map f t).prod
    -/
    obtain rfl | hm := m.eq_zero_or_pos
      /-
        case inl
        α : Type u_2
        β : Type u_3
        inst✝¹ : CommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        n : Nat
        hf : IsMulFreimanHom n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        h : Eq s.prod t.prod
        hmn : LE.le 0 n
        hs : Eq s.card 0
        ht : Eq t.card 0
        ⊢ Eq (Multiset.map f s).prod (Multiset.map f t).prod
      -/
    · rw [card_eq_zero] at hs ht
      /-
        case inl
        α : Type u_2
        β : Type u_3
        inst✝¹ : CommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        n : Nat
        hf : IsMulFreimanHom n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        h : Eq s.prod t.prod
        hmn : LE.le 0 n
        hs : Eq s 0
        ht : Eq t 0
        ⊢ Eq (Multiset.map f s).prod (Multiset.map f t).prod
      -/
      rw [hs, ht]
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u_2
      β : Type u_3
      inst✝¹ : CommMonoid α
      inst✝ : CancelCommMonoid β
      A : Set α
      B : Set β
      f : α → β
      m n : Nat
      hmn : LE.le m n
      hf : IsMulFreimanHom n A B f
      s t : Multiset α
      hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
      htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
      hs : Eq s.card m
      ht : Eq t.card m
      h : Eq s.prod t.prod
      hm : GT.gt m 0
      ⊢ Eq (Multiset.map f s).prod (Multiset.map f t).prod
    -/
    simp only [← hs, card_pos_iff_exists_mem] at hm
    /-
      case inr
      α : Type u_2
      β : Type u_3
      inst✝¹ : CommMonoid α
      inst✝ : CancelCommMonoid β
      A : Set α
      B : Set β
      f : α → β
      m n : Nat
      hmn : LE.le m n
      hf : IsMulFreimanHom n A B f
      s t : Multiset α
      hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
      htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
      hs : Eq s.card m
      ht : Eq t.card m
      h : Eq s.prod t.prod
      hm : Exists fun a => Membership.mem s a
      ⊢ Eq (Multiset.map f s).prod (Multiset.map f t).prod
    -/
    obtain ⟨a, ha⟩ := hm
    suffices ((s + replicate (n - m) a).map f).prod = ((t + replicate (n - m) a).map f).prod by
      simp_rw [Multiset.map_add, prod_add] at this
      exact mul_right_cancel this
    /-
      case inr.intro
      α : Type u_2
      β : Type u_3
      inst✝¹ : CommMonoid α
      inst✝ : CancelCommMonoid β
      A : Set α
      B : Set β
      f : α → β
      m n : Nat
      hmn : LE.le m n
      hf : IsMulFreimanHom n A B f
      s t : Multiset α
      hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
      htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
      hs : Eq s.card m
      ht : Eq t.card m
      h : Eq s.prod t.prod
      a : α
      ha : Membership.mem s a
      ⊢ Eq (Multiset.map f (HAdd.hAdd s (Multiset.replicate (HSub.hSub n m) a))).pro …
    -/
    replace ha := hsA ha
    /-
      case inr.intro
      α : Type u_2
      β : Type u_3
      inst✝¹ : CommMonoid α
      inst✝ : CancelCommMonoid β
      A : Set α
      B : Set β
      f : α → β
      m n : Nat
      hmn : LE.le m n
      hf : IsMulFreimanHom n A B f
      s t : Multiset α
      hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
      htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
      hs : Eq s.card m
      ht : Eq t.card m
      h : Eq s.prod t.prod
      a : α
      ha : Membership.mem A a
      ⊢ Eq (Multiset.map f (HAdd.hAdd s (Multiset.replicate (HSub.hSub n m) a))).pro …
    -/
    refine hf.map_prod_eq_map_prod (fun a ha ↦ ?_) (fun a ha ↦ ?_) ?_ ?_ ?_
      /-
        case inr.intro.refine_1
        α : Type u_2
        β : Type u_3
        inst✝¹ : CommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        m n : Nat
        hmn : LE.le m n
        hf : IsMulFreimanHom n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card m
        ht : Eq t.card m
        h : Eq s.prod t.prod
        a✝ : α
        ha✝ : Membership.mem A a✝
        a : α
        ha : Membership.mem (HAdd.hAdd s (Multiset.replicate (HSub.hSub n m) a✝)) a
        ⊢ Membership.mem A a
      -/
    · rw [Multiset.mem_add] at ha
      /-
        case inr.intro.refine_1
        α : Type u_2
        β : Type u_3
        inst✝¹ : CommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        m n : Nat
        hmn : LE.le m n
        hf : IsMulFreimanHom n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card m
        ht : Eq t.card m
        h : Eq s.prod t.prod
        a✝ : α
        ha✝ : Membership.mem A a✝
        a : α
        ha : Or (Membership.mem s a) (Membership.mem (Multiset.replicate (HSub.hSub n  …
        ⊢ Membership.mem A a
      -/
      obtain ha | ha := ha
        /-
          case inr.intro.refine_1.inl
          α : Type u_2
          β : Type u_3
          inst✝¹ : CommMonoid α
          inst✝ : CancelCommMonoid β
          A : Set α
          B : Set β
          f : α → β
          m n : Nat
          hmn : LE.le m n
          hf : IsMulFreimanHom n A B f
          s t : Multiset α
          hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
          htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
          hs : Eq s.card m
          ht : Eq t.card m
          h : Eq s.prod t.prod
          a✝ : α
          ha✝ : Membership.mem A a✝
          a : α
          ha : Membership.mem s a
          ⊢ Membership.mem A a
        -/
      · exact hsA ha
        /-
          🎉 no goals
        -/
        /-
          case inr.intro.refine_1.inr
          α : Type u_2
          β : Type u_3
          inst✝¹ : CommMonoid α
          inst✝ : CancelCommMonoid β
          A : Set α
          B : Set β
          f : α → β
          m n : Nat
          hmn : LE.le m n
          hf : IsMulFreimanHom n A B f
          s t : Multiset α
          hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
          htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
          hs : Eq s.card m
          ht : Eq t.card m
          h : Eq s.prod t.prod
          a✝ : α
          ha✝ : Membership.mem A a✝
          a : α
          ha : Membership.mem (Multiset.replicate (HSub.hSub n m) a✝) a
          ⊢ Membership.mem A a
        -/
      · rwa [eq_of_mem_replicate ha]
        /-
          🎉 no goals
        -/
      /-
        case inr.intro.refine_2
        α : Type u_2
        β : Type u_3
        inst✝¹ : CommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        m n : Nat
        hmn : LE.le m n
        hf : IsMulFreimanHom n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card m
        ht : Eq t.card m
        h : Eq s.prod t.prod
        a✝ : α
        ha✝ : Membership.mem A a✝
        a : α
        ha : Membership.mem (HAdd.hAdd t (Multiset.replicate (HSub.hSub n m) a✝)) a
        ⊢ Membership.mem A a
      -/
    · rw [Multiset.mem_add] at ha
      /-
        case inr.intro.refine_2
        α : Type u_2
        β : Type u_3
        inst✝¹ : CommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        m n : Nat
        hmn : LE.le m n
        hf : IsMulFreimanHom n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card m
        ht : Eq t.card m
        h : Eq s.prod t.prod
        a✝ : α
        ha✝ : Membership.mem A a✝
        a : α
        ha : Or (Membership.mem t a) (Membership.mem (Multiset.replicate (HSub.hSub n  …
        ⊢ Membership.mem A a
      -/
      obtain ha | ha := ha
        /-
          case inr.intro.refine_2.inl
          α : Type u_2
          β : Type u_3
          inst✝¹ : CommMonoid α
          inst✝ : CancelCommMonoid β
          A : Set α
          B : Set β
          f : α → β
          m n : Nat
          hmn : LE.le m n
          hf : IsMulFreimanHom n A B f
          s t : Multiset α
          hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
          htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
          hs : Eq s.card m
          ht : Eq t.card m
          h : Eq s.prod t.prod
          a✝ : α
          ha✝ : Membership.mem A a✝
          a : α
          ha : Membership.mem t a
          ⊢ Membership.mem A a
        -/
      · exact htA ha
        /-
          🎉 no goals
        -/
        /-
          case inr.intro.refine_2.inr
          α : Type u_2
          β : Type u_3
          inst✝¹ : CommMonoid α
          inst✝ : CancelCommMonoid β
          A : Set α
          B : Set β
          f : α → β
          m n : Nat
          hmn : LE.le m n
          hf : IsMulFreimanHom n A B f
          s t : Multiset α
          hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
          htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
          hs : Eq s.card m
          ht : Eq t.card m
          h : Eq s.prod t.prod
          a✝ : α
          ha✝ : Membership.mem A a✝
          a : α
          ha : Membership.mem (Multiset.replicate (HSub.hSub n m) a✝) a
          ⊢ Membership.mem A a
        -/
      · rwa [eq_of_mem_replicate ha]
        /-
          🎉 no goals
        -/
      /-
        case inr.intro.refine_3
        α : Type u_2
        β : Type u_3
        inst✝¹ : CommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        m n : Nat
        hmn : LE.le m n
        hf : IsMulFreimanHom n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card m
        ht : Eq t.card m
        h : Eq s.prod t.prod
        a : α
        ha : Membership.mem A a
        ⊢ Eq (HAdd.hAdd s (Multiset.replicate (HSub.hSub n m) a)).card n
      -/
    · rw [card_add, card_replicate, hs, Nat.add_sub_cancel' hmn]
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.refine_4
        α : Type u_2
        β : Type u_3
        inst✝¹ : CommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        m n : Nat
        hmn : LE.le m n
        hf : IsMulFreimanHom n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card m
        ht : Eq t.card m
        h : Eq s.prod t.prod
        a : α
        ha : Membership.mem A a
        ⊢ Eq (HAdd.hAdd t (Multiset.replicate (HSub.hSub n m) a)).card n
      -/
    · rw [card_add, card_replicate, ht, Nat.add_sub_cancel' hmn]
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.refine_5
        α : Type u_2
        β : Type u_3
        inst✝¹ : CommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        m n : Nat
        hmn : LE.le m n
        hf : IsMulFreimanHom n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card m
        ht : Eq t.card m
        h : Eq s.prod t.prod
        a : α
        ha : Membership.mem A a
        ⊢ Eq (HAdd.hAdd s (Multiset.replicate (HSub.hSub n m) a)).prod (HAdd.hAdd t (M …
      -/
    · rw [prod_add, prod_add, h]
      /-
        🎉 no goals
      -/


@[to_additive]
lemma IsMulFreimanIso.mono {hmn : m ≤ n} (hf : IsMulFreimanIso n A B f) :
    IsMulFreimanIso m A B f where
  bijOn := hf.bijOn
  map_prod_eq_map_prod s t hsA htA hs ht := by
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : CancelCommMonoid α
      inst✝ : CancelCommMonoid β
      A : Set α
      B : Set β
      f : α → β
      m n : Nat
      hmn : LE.le m n
      hf : IsMulFreimanIso n A B f
      s t : Multiset α
      hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
      htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
      hs : Eq s.card m
      ht : Eq t.card m
      ⊢ Iff (Eq (Multiset.map f s).prod (Multiset.map f t).prod) (Eq s.prod t.prod)
    -/
    obtain rfl | hm := m.eq_zero_or_pos
      /-
        case inl
        α : Type u_2
        β : Type u_3
        inst✝¹ : CancelCommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        n : Nat
        hf : IsMulFreimanIso n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hmn : LE.le 0 n
        hs : Eq s.card 0
        ht : Eq t.card 0
        ⊢ Iff (Eq (Multiset.map f s).prod (Multiset.map f t).prod) (Eq s.prod t.prod)
      -/
    · rw [card_eq_zero] at hs ht
      /-
        case inl
        α : Type u_2
        β : Type u_3
        inst✝¹ : CancelCommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        n : Nat
        hf : IsMulFreimanIso n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hmn : LE.le 0 n
        hs : Eq s 0
        ht : Eq t 0
        ⊢ Iff (Eq (Multiset.map f s).prod (Multiset.map f t).prod) (Eq s.prod t.prod)
      -/
      simp [hs, ht]
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u_2
      β : Type u_3
      inst✝¹ : CancelCommMonoid α
      inst✝ : CancelCommMonoid β
      A : Set α
      B : Set β
      f : α → β
      m n : Nat
      hmn : LE.le m n
      hf : IsMulFreimanIso n A B f
      s t : Multiset α
      hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
      htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
      hs : Eq s.card m
      ht : Eq t.card m
      hm : GT.gt m 0
      ⊢ Iff (Eq (Multiset.map f s).prod (Multiset.map f t).prod) (Eq s.prod t.prod)
    -/
    simp only [← hs, card_pos_iff_exists_mem] at hm
    /-
      case inr
      α : Type u_2
      β : Type u_3
      inst✝¹ : CancelCommMonoid α
      inst✝ : CancelCommMonoid β
      A : Set α
      B : Set β
      f : α → β
      m n : Nat
      hmn : LE.le m n
      hf : IsMulFreimanIso n A B f
      s t : Multiset α
      hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
      htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
      hs : Eq s.card m
      ht : Eq t.card m
      hm : Exists fun a => Membership.mem s a
      ⊢ Iff (Eq (Multiset.map f s).prod (Multiset.map f t).prod) (Eq s.prod t.prod)
    -/
    obtain ⟨a, ha⟩ := hm
    suffices
      ((s + replicate (n - m) a).map f).prod = ((t + replicate (n - m) a).map f).prod ↔
      (s + replicate (n - m) a).prod = (t + replicate (n - m) a).prod by
      simpa only [Multiset.map_add, prod_add, mul_right_cancel_iff] using this
    /-
      case inr.intro
      α : Type u_2
      β : Type u_3
      inst✝¹ : CancelCommMonoid α
      inst✝ : CancelCommMonoid β
      A : Set α
      B : Set β
      f : α → β
      m n : Nat
      hmn : LE.le m n
      hf : IsMulFreimanIso n A B f
      s t : Multiset α
      hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
      htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
      hs : Eq s.card m
      ht : Eq t.card m
      a : α
      ha : Membership.mem s a
      ⊢ Iff (Eq (Multiset.map f (HAdd.hAdd s (Multiset.replicate (HSub.hSub n m) a)) …
    -/
    replace ha := hsA ha
    /-
      case inr.intro
      α : Type u_2
      β : Type u_3
      inst✝¹ : CancelCommMonoid α
      inst✝ : CancelCommMonoid β
      A : Set α
      B : Set β
      f : α → β
      m n : Nat
      hmn : LE.le m n
      hf : IsMulFreimanIso n A B f
      s t : Multiset α
      hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
      htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
      hs : Eq s.card m
      ht : Eq t.card m
      a : α
      ha : Membership.mem A a
      ⊢ Iff (Eq (Multiset.map f (HAdd.hAdd s (Multiset.replicate (HSub.hSub n m) a)) …
    -/
    refine hf.map_prod_eq_map_prod (fun a ha ↦ ?_) (fun a ha ↦ ?_) ?_ ?_
      /-
        case inr.intro.refine_1
        α : Type u_2
        β : Type u_3
        inst✝¹ : CancelCommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        m n : Nat
        hmn : LE.le m n
        hf : IsMulFreimanIso n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card m
        ht : Eq t.card m
        a✝ : α
        ha✝ : Membership.mem A a✝
        a : α
        ha : Membership.mem (HAdd.hAdd s (Multiset.replicate (HSub.hSub n m) a✝)) a
        ⊢ Membership.mem A a
      -/
    · rw [Multiset.mem_add] at ha
      /-
        case inr.intro.refine_1
        α : Type u_2
        β : Type u_3
        inst✝¹ : CancelCommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        m n : Nat
        hmn : LE.le m n
        hf : IsMulFreimanIso n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card m
        ht : Eq t.card m
        a✝ : α
        ha✝ : Membership.mem A a✝
        a : α
        ha : Or (Membership.mem s a) (Membership.mem (Multiset.replicate (HSub.hSub n  …
        ⊢ Membership.mem A a
      -/
      obtain ha | ha := ha
        /-
          case inr.intro.refine_1.inl
          α : Type u_2
          β : Type u_3
          inst✝¹ : CancelCommMonoid α
          inst✝ : CancelCommMonoid β
          A : Set α
          B : Set β
          f : α → β
          m n : Nat
          hmn : LE.le m n
          hf : IsMulFreimanIso n A B f
          s t : Multiset α
          hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
          htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
          hs : Eq s.card m
          ht : Eq t.card m
          a✝ : α
          ha✝ : Membership.mem A a✝
          a : α
          ha : Membership.mem s a
          ⊢ Membership.mem A a
        -/
      · exact hsA ha
        /-
          🎉 no goals
        -/
        /-
          case inr.intro.refine_1.inr
          α : Type u_2
          β : Type u_3
          inst✝¹ : CancelCommMonoid α
          inst✝ : CancelCommMonoid β
          A : Set α
          B : Set β
          f : α → β
          m n : Nat
          hmn : LE.le m n
          hf : IsMulFreimanIso n A B f
          s t : Multiset α
          hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
          htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
          hs : Eq s.card m
          ht : Eq t.card m
          a✝ : α
          ha✝ : Membership.mem A a✝
          a : α
          ha : Membership.mem (Multiset.replicate (HSub.hSub n m) a✝) a
          ⊢ Membership.mem A a
        -/
      · rwa [eq_of_mem_replicate ha]
        /-
          🎉 no goals
        -/
      /-
        case inr.intro.refine_2
        α : Type u_2
        β : Type u_3
        inst✝¹ : CancelCommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        m n : Nat
        hmn : LE.le m n
        hf : IsMulFreimanIso n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card m
        ht : Eq t.card m
        a✝ : α
        ha✝ : Membership.mem A a✝
        a : α
        ha : Membership.mem (HAdd.hAdd t (Multiset.replicate (HSub.hSub n m) a✝)) a
        ⊢ Membership.mem A a
      -/
    · rw [Multiset.mem_add] at ha
      /-
        case inr.intro.refine_2
        α : Type u_2
        β : Type u_3
        inst✝¹ : CancelCommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        m n : Nat
        hmn : LE.le m n
        hf : IsMulFreimanIso n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card m
        ht : Eq t.card m
        a✝ : α
        ha✝ : Membership.mem A a✝
        a : α
        ha : Or (Membership.mem t a) (Membership.mem (Multiset.replicate (HSub.hSub n  …
        ⊢ Membership.mem A a
      -/
      obtain ha | ha := ha
        /-
          case inr.intro.refine_2.inl
          α : Type u_2
          β : Type u_3
          inst✝¹ : CancelCommMonoid α
          inst✝ : CancelCommMonoid β
          A : Set α
          B : Set β
          f : α → β
          m n : Nat
          hmn : LE.le m n
          hf : IsMulFreimanIso n A B f
          s t : Multiset α
          hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
          htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
          hs : Eq s.card m
          ht : Eq t.card m
          a✝ : α
          ha✝ : Membership.mem A a✝
          a : α
          ha : Membership.mem t a
          ⊢ Membership.mem A a
        -/
      · exact htA ha
        /-
          🎉 no goals
        -/
        /-
          case inr.intro.refine_2.inr
          α : Type u_2
          β : Type u_3
          inst✝¹ : CancelCommMonoid α
          inst✝ : CancelCommMonoid β
          A : Set α
          B : Set β
          f : α → β
          m n : Nat
          hmn : LE.le m n
          hf : IsMulFreimanIso n A B f
          s t : Multiset α
          hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
          htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
          hs : Eq s.card m
          ht : Eq t.card m
          a✝ : α
          ha✝ : Membership.mem A a✝
          a : α
          ha : Membership.mem (Multiset.replicate (HSub.hSub n m) a✝) a
          ⊢ Membership.mem A a
        -/
      · rwa [eq_of_mem_replicate ha]
        /-
          🎉 no goals
        -/
      /-
        case inr.intro.refine_3
        α : Type u_2
        β : Type u_3
        inst✝¹ : CancelCommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        m n : Nat
        hmn : LE.le m n
        hf : IsMulFreimanIso n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card m
        ht : Eq t.card m
        a : α
        ha : Membership.mem A a
        ⊢ Eq (HAdd.hAdd s (Multiset.replicate (HSub.hSub n m) a)).card n
      -/
    · rw [card_add, card_replicate, hs, Nat.add_sub_cancel' hmn]
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.refine_4
        α : Type u_2
        β : Type u_3
        inst✝¹ : CancelCommMonoid α
        inst✝ : CancelCommMonoid β
        A : Set α
        B : Set β
        f : α → β
        m n : Nat
        hmn : LE.le m n
        hf : IsMulFreimanIso n A B f
        s t : Multiset α
        hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
        htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
        hs : Eq s.card m
        ht : Eq t.card m
        a : α
        ha : Membership.mem A a
        ⊢ Eq (HAdd.hAdd t (Multiset.replicate (HSub.hSub n m) a)).card n
      -/
    · rw [card_add, card_replicate, ht, Nat.add_sub_cancel' hmn]
      /-
        🎉 no goals
      -/


@[to_additive]
lemma IsMulFreimanHom.inv (hf : IsMulFreimanHom n A B f) : IsMulFreimanHom n A B⁻¹ f⁻¹ where
  mapsTo := hf.mapsTo.inv
  map_prod_eq_map_prod s t hsA htA hs ht h := by
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : CommMonoid α
      inst✝ : DivisionCommMonoid β
      A : Set α
      B : Set β
      f : α → β
      n : Nat
      hf : IsMulFreimanHom n A B f
      s t : Multiset α
      hsA : ∀ ⦃x : α⦄, Membership.mem s x → Membership.mem A x
      htA : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem A x
      hs : Eq s.card n
      ht : Eq t.card n
      h : Eq s.prod t.prod
      ⊢ Eq (Multiset.map (Inv.inv f) s).prod (Multiset.map (Inv.inv f) t).prod
    -/
    rw [Pi.inv_def, prod_map_inv, prod_map_inv, hf.map_prod_eq_map_prod hsA htA hs ht h]
    /-
      🎉 no goals
    -/


@[to_additive] lemma IsMulFreimanHom.div {β : Type*} [DivisionCommMonoid β] {B₁ B₂ : Set β}
    {f₁ f₂ : α → β} (h₁ : IsMulFreimanHom n A B₁ f₁) (h₂ : IsMulFreimanHom n A B₂ f₂) :
    IsMulFreimanHom n A (B₁ / B₂) (f₁ / f₂) where
  mapsTo := h₁.mapsTo.div h₂.mapsTo
  map_prod_eq_map_prod s t hsA htA hs ht h := by
    rw [Pi.div_def, prod_map_div, prod_map_div, h₁.map_prod_eq_map_prod hsA htA hs ht h,
      h₂.map_prod_eq_map_prod hsA htA hs ht h]


@[to_additive]
lemma IsMulFreimanHom.prod (h₁ : IsMulFreimanHom n A₁ B₁ f₁) (h₂ : IsMulFreimanHom n A₂ B₂ f₂) :
    IsMulFreimanHom n (A₁ ×ˢ A₂) (B₁ ×ˢ B₂) (Prod.map f₁ f₂) where
  mapsTo := h₁.mapsTo.prodMap h₂.mapsTo
  map_prod_eq_map_prod s t hsA htA hs ht h := by
    /-
      α₁ : Type u_5
      α₂ : Type u_6
      β₁ : Type u_7
      β₂ : Type u_8
      inst✝³ : CommMonoid α₁
      inst✝² : CommMonoid α₂
      inst✝¹ : CommMonoid β₁
      inst✝ : CommMonoid β₂
      A₁ : Set α₁
      A₂ : Set α₂
      B₁ : Set β₁
      B₂ : Set β₂
      f₁ : α₁ → β₁
      f₂ : α₂ → β₂
      n : Nat
      h₁ : IsMulFreimanHom n A₁ B₁ f₁
      h₂ : IsMulFreimanHom n A₂ B₂ f₂
      s t : Multiset (Prod α₁ α₂)
      hsA : ∀ ⦃x : Prod α₁ α₂⦄, Membership.mem s x → Membership.mem (SProd.sprod A₁  …
      htA : ∀ ⦃x : Prod α₁ α₂⦄, Membership.mem t x → Membership.mem (SProd.sprod A₁  …
      hs : Eq s.card n
      ht : Eq t.card n
      h : Eq s.prod t.prod
      ⊢ Eq (Multiset.map (Prod.map f₁ f₂) s).prod (Multiset.map (Prod.map f₁ f₂) t). …
    -/
    simp only [mem_prod, forall_and, Prod.forall] at hsA htA
    simp only [Prod.ext_iff, fst_prod, snd_prod, map_map, Function.comp_apply, Prod.map_fst,
      Prod.map_snd] at h ⊢
    /-
      α₁ : Type u_5
      α₂ : Type u_6
      β₁ : Type u_7
      β₂ : Type u_8
      inst✝³ : CommMonoid α₁
      inst✝² : CommMonoid α₂
      inst✝¹ : CommMonoid β₁
      inst✝ : CommMonoid β₂
      A₁ : Set α₁
      A₂ : Set α₂
      B₁ : Set β₁
      B₂ : Set β₂
      f₁ : α₁ → β₁
      f₂ : α₂ → β₂
      n : Nat
      h₁ : IsMulFreimanHom n A₁ B₁ f₁
      h₂ : IsMulFreimanHom n A₂ B₂ f₂
      s t : Multiset (Prod α₁ α₂)
      hs : Eq s.card n
      ht : Eq t.card n
      hsA : And (∀ (a : α₁) (b : α₂), Membership.mem s { fst := a, snd := b } → Memb …
      htA : And (∀ (a : α₁) (b : α₂), Membership.mem t { fst := a, snd := b } → Memb …
      h : And (Eq (Multiset.map Prod.fst s).prod (Multiset.map Prod.fst t).prod) (Eq …
      ⊢ And (Eq (Multiset.map (fun x => f₁ x.1) s).prod (Multiset.map (fun x => f₁ x …
    -/
    rw [← Function.comp_def, ← map_map, ← map_map, ← Function.comp_def f₂, ← map_map, ← map_map]
    exact ⟨h₁.map_prod_eq_map_prod (by simpa using hsA.1) (by simpa using htA.1) (by simpa)
      (by simpa) h.1, h₂.map_prod_eq_map_prod (by simpa [@forall_swap α₁] using hsA.2)
      (by simpa [@forall_swap α₁] using htA.2) (by simpa) (by simpa) h.2⟩


@[to_additive]
lemma IsMulFreimanIso.prod (h₁ : IsMulFreimanIso n A₁ B₁ f₁) (h₂ : IsMulFreimanIso n A₂ B₂ f₂) :
    IsMulFreimanIso n (A₁ ×ˢ A₂) (B₁ ×ˢ B₂) (Prod.map f₁ f₂) where
  bijOn := h₁.bijOn.prodMap h₂.bijOn
  map_prod_eq_map_prod s t hsA htA hs ht := by
    /-
      α₁ : Type u_5
      α₂ : Type u_6
      β₁ : Type u_7
      β₂ : Type u_8
      inst✝³ : CommMonoid α₁
      inst✝² : CommMonoid α₂
      inst✝¹ : CommMonoid β₁
      inst✝ : CommMonoid β₂
      A₁ : Set α₁
      A₂ : Set α₂
      B₁ : Set β₁
      B₂ : Set β₂
      f₁ : α₁ → β₁
      f₂ : α₂ → β₂
      n : Nat
      h₁ : IsMulFreimanIso n A₁ B₁ f₁
      h₂ : IsMulFreimanIso n A₂ B₂ f₂
      s t : Multiset (Prod α₁ α₂)
      hsA : ∀ ⦃x : Prod α₁ α₂⦄, Membership.mem s x → Membership.mem (SProd.sprod A₁  …
      htA : ∀ ⦃x : Prod α₁ α₂⦄, Membership.mem t x → Membership.mem (SProd.sprod A₁  …
      hs : Eq s.card n
      ht : Eq t.card n
      ⊢ Iff (Eq (Multiset.map (Prod.map f₁ f₂) s).prod (Multiset.map (Prod.map f₁ f₂ …
    -/
    simp only [mem_prod, forall_and, Prod.forall] at hsA htA
    simp only [Prod.ext_iff, fst_prod, map_map, Function.comp_apply, Prod.map_fst, snd_prod,
      Prod.map_snd]
    rw [← Function.comp_def, ← map_map, ← map_map, ← Function.comp_def f₂, ← map_map, ← map_map,
      h₁.map_prod_eq_map_prod (by simpa using hsA.1) (by simpa using htA.1) (by simpa) (by simpa),
      h₂.map_prod_eq_map_prod (by simpa [@forall_swap α₁] using hsA.2)
      (by simpa [@forall_swap α₁] using htA.2) (by simpa) (by simpa)]


private lemma aux (hm : m ≠ 0) (hkmn : m * k ≤ n) : k < (n + 1) :=
  Nat.lt_succ_iff.2 <| le_trans (Nat.le_mul_of_pos_left _ hm.bot_lt) hkmn


/-- **No wrap-around principle**.

The first `k + 1` elements of `Fin (n + 1)` are `m`-Freiman isomorphic to the first `k + 1` elements
of `ℕ` assuming there is no wrap-around. -/
lemma isAddFreimanIso_Iic (hm : m ≠ 0) (hkmn : m * k ≤ n) :
    IsAddFreimanIso m (Iic (k : Fin (n + 1))) (Iic k) val where
                   /-
                     k m n : Nat
                     hm : Ne m 0
                     hkmn : LE.le (HMul.hMul m k) n
                     ⊢ Set.MapsTo Fin.val (Set.Iic ↑k) (Set.Iic k)
                   -/
  bijOn.left := by simp [MapsTo, Fin.le_iff_val_le_val, Nat.mod_eq_of_lt, aux hm hkmn]
                   /-
                     🎉 no goals
                   -/
  bijOn.right.left := val_injective.injOn
  bijOn.right.right x (hx : x ≤ _) :=
           /-
             k m n : Nat
             hm : Ne m 0
             hkmn : LE.le (HMul.hMul m k) n
             x : Nat
             hx : LE.le x k
             ⊢ And (Membership.mem (Set.Iic ↑k) ↑x) (Eq (↑↑x) x)
           -/
    ⟨x, by simpa [le_iff_val_le_val, -val_fin_le, Nat.mod_eq_of_lt, aux hm hkmn, hx.trans_lt]⟩
           /-
             🎉 no goals
           -/
  map_sum_eq_map_sum s t hsA htA hs ht := by
    /-
      k m n : Nat
      hm : Ne m 0
      hkmn : LE.le (HMul.hMul m k) n
      s t : Multiset (Fin (HAdd.hAdd n 1))
      hsA : ∀ ⦃x : Fin (HAdd.hAdd n 1)⦄, Membership.mem s x → Membership.mem (Set.Ii …
      htA : ∀ ⦃x : Fin (HAdd.hAdd n 1)⦄, Membership.mem t x → Membership.mem (Set.Ii …
      hs : Eq s.card m
      ht : Eq t.card m
      ⊢ Iff (Eq (Multiset.map Fin.val s).sum (Multiset.map Fin.val t).sum) (Eq s.sum …
    -/
    have (u : Multiset (Fin (n + 1))) : Nat.castRingHom _ (u.map val).sum = u.sum := by simp
    /-
      k m n : Nat
      hm : Ne m 0
      hkmn : LE.le (HMul.hMul m k) n
      s t : Multiset (Fin (HAdd.hAdd n 1))
      hsA : ∀ ⦃x : Fin (HAdd.hAdd n 1)⦄, Membership.mem s x → Membership.mem (Set.Ii …
      htA : ∀ ⦃x : Fin (HAdd.hAdd n 1)⦄, Membership.mem t x → Membership.mem (Set.Ii …
      hs : Eq s.card m
      ht : Eq t.card m
      this : ∀ (u : Multiset (Fin (HAdd.hAdd n 1))), Eq ((Nat.castRingHom (Fin (HAdd …
      ⊢ Iff (Eq (Multiset.map Fin.val s).sum (Multiset.map Fin.val t).sum) (Eq s.sum …
    -/
    rw [← this, ← this]
    have {u : Multiset (Fin (n + 1))} (huk : ∀ x ∈ u, x ≤ k) (hu : card u = m) :
        (u.map val).sum < (n + 1) := Nat.lt_succ_iff.2 <| hkmn.trans' <| by
      rw [← hu, ← card_map]
      refine sum_le_card_nsmul (u.map val) k ?_
      simpa [le_iff_val_le_val, -val_fin_le, Nat.mod_eq_of_lt, aux hm hkmn] using huk
    /-
      k m n : Nat
      hm : Ne m 0
      hkmn : LE.le (HMul.hMul m k) n
      s t : Multiset (Fin (HAdd.hAdd n 1))
      hsA : ∀ ⦃x : Fin (HAdd.hAdd n 1)⦄, Membership.mem s x → Membership.mem (Set.Ii …
      htA : ∀ ⦃x : Fin (HAdd.hAdd n 1)⦄, Membership.mem t x → Membership.mem (Set.Ii …
      hs : Eq s.card m
      ht : Eq t.card m
      this✝ : ∀ (u : Multiset (Fin (HAdd.hAdd n 1))), Eq ((Nat.castRingHom (Fin (HAd …
      this : ∀ {u : Multiset (Fin (HAdd.hAdd n 1))}, (∀ (x : Fin (HAdd.hAdd n 1)), M …
      ⊢ Iff (Eq (Multiset.map Fin.val s).sum (Multiset.map Fin.val t).sum) (Eq ((Nat …
    -/
    exact ⟨congr_arg _, CharP.natCast_injOn_Iio _ (n + 1) (this hsA hs) (this htA ht)⟩
    /-
      🎉 no goals
    -/


/-- **No wrap-around principle**.

The first `k` elements of `Fin (n + 1)` are `m`-Freiman isomorphic to the first `k` elements of `ℕ`
assuming there is no wrap-around. -/
lemma isAddFreimanIso_Iio (hm : m ≠ 0) (hkmn : m * k ≤ n) :
    IsAddFreimanIso m (Iio (k : Fin (n + 1))) (Iio k) val := by
  /-
    k m n : Nat
    hm : Ne m 0
    hkmn : LE.le (HMul.hMul m k) n
    ⊢ IsAddFreimanIso m (Set.Iio ↑k) (Set.Iio k) Fin.val
  -/
  obtain _ | k := k
    /-
      case zero
      m n : Nat
      hm : Ne m 0
      hkmn : LE.le (HMul.hMul m 0) n
      ⊢ IsAddFreimanIso m (Set.Iio ↑0) (Set.Iio 0) Fin.val
    -/
  · simp [← bot_eq_zero]; simp [← _root_.bot_eq_zero, -Nat.bot_eq_zero, -bot_eq_zero']
                          /-
                            🎉 no goals
                          -/
  /-
    case succ
    m n : Nat
    hm : Ne m 0
    k : Nat
    hkmn : LE.le (HMul.hMul m (HAdd.hAdd k 1)) n
    ⊢ IsAddFreimanIso m (Set.Iio ↑(HAdd.hAdd k 1)) (Set.Iio (HAdd.hAdd k 1)) Fin.val
  -/
  have hkmn' : m * k ≤ n := (Nat.mul_le_mul_left _ k.le_succ).trans hkmn
  /-
    case succ
    m n : Nat
    hm : Ne m 0
    k : Nat
    hkmn : LE.le (HMul.hMul m (HAdd.hAdd k 1)) n
    hkmn' : LE.le (HMul.hMul m k) n
    ⊢ IsAddFreimanIso m (Set.Iio ↑(HAdd.hAdd k 1)) (Set.Iio (HAdd.hAdd k 1)) Fin.val
  -/
  convert isAddFreimanIso_Iic hm hkmn' using 1 <;> ext x
  · simp [lt_iff_val_lt_val, le_iff_val_le_val, -val_fin_le, -val_fin_lt, Nat.mod_eq_of_lt,
      aux hm hkmn']
    /-
      case h.e'_6.h
      m n : Nat
      hm : Ne m 0
      k : Nat
      hkmn : LE.le (HMul.hMul m (HAdd.hAdd k 1)) n
      hkmn' : LE.le (HMul.hMul m k) n
      x : Fin (HAdd.hAdd n 1)
      ⊢ Iff (LT.lt ↑x ↑(HAdd.hAdd (↑k) 1)) (LE.le (↑x) k)
    -/
    simp_rw [← Nat.cast_add_one]
    /-
      case h.e'_6.h
      m n : Nat
      hm : Ne m 0
      k : Nat
      hkmn : LE.le (HMul.hMul m (HAdd.hAdd k 1)) n
      hkmn' : LE.le (HMul.hMul m k) n
      x : Fin (HAdd.hAdd n 1)
      ⊢ Iff (LT.lt ↑x ↑↑(HAdd.hAdd k 1)) (LE.le (↑x) k)
    -/
    rw [Fin.val_cast_of_lt (aux hm hkmn), Nat.lt_succ_iff]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_7.h
      m n : Nat
      hm : Ne m 0
      k : Nat
      hkmn : LE.le (HMul.hMul m (HAdd.hAdd k 1)) n
      hkmn' : LE.le (HMul.hMul m k) n
      x : Nat
      ⊢ Iff (Membership.mem (Set.Iio (HAdd.hAdd k 1)) x) (Membership.mem (Set.Iic k) …
    -/
  · simp [Nat.lt_succ_iff]
    /-
      🎉 no goals
    -/


