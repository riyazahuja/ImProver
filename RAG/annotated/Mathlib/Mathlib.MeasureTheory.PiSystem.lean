/-- A π-system is a collection of subsets of `α` that is closed under binary intersection of
  non-disjoint sets. Usually it is also required that the collection is nonempty, but we don't do
  that here. -/
def IsPiSystem (C : Set (Set α)) : Prop :=
  ∀ᵉ (s ∈ C) (t ∈ C), (s ∩ t : Set α).Nonempty → s ∩ t ∈ C


theorem isPiSystem_measurableSet {α : Type*} [MeasurableSpace α] :
    IsPiSystem { s : Set α | MeasurableSet s } := fun _ hs _ ht _ => hs.inter ht


theorem IsPiSystem.singleton (S : Set α) : IsPiSystem ({S} : Set (Set α)) := by
  /-
    α : Type u_1
    S : Set α
    ⊢ IsPiSystem (Singleton.singleton S)
  -/
  intro s h_s t h_t _
  rw [Set.mem_singleton_iff.1 h_s, Set.mem_singleton_iff.1 h_t, Set.inter_self,
    Set.mem_singleton_iff]


theorem IsPiSystem.insert_empty {S : Set (Set α)} (h_pi : IsPiSystem S) :
    IsPiSystem (insert ∅ S) := by
  /-
    α : Type u_1
    S : Set (Set α)
    h_pi : IsPiSystem S
    ⊢ IsPiSystem (Insert.insert EmptyCollection.emptyCollection S)
  -/
  intro s hs t ht hst
  /-
    α : Type u_1
    S : Set (Set α)
    h_pi : IsPiSystem S
    s : Set α
    hs : Membership.mem (Insert.insert EmptyCollection.emptyCollection S) s
    t : Set α
    ht : Membership.mem (Insert.insert EmptyCollection.emptyCollection S) t
    hst : (Inter.inter s t).Nonempty
    ⊢ Membership.mem (Insert.insert EmptyCollection.emptyCollection S) (Inter.inte …
  -/
  cases' hs with hs hs
    /-
      case inl
      α : Type u_1
      S : Set (Set α)
      h_pi : IsPiSystem S
      s t : Set α
      ht : Membership.mem (Insert.insert EmptyCollection.emptyCollection S) t
      hst : (Inter.inter s t).Nonempty
      hs : Eq s EmptyCollection.emptyCollection
      ⊢ Membership.mem (Insert.insert EmptyCollection.emptyCollection S) (Inter.inte …
    -/
  · simp [hs]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      S : Set (Set α)
      h_pi : IsPiSystem S
      s t : Set α
      ht : Membership.mem (Insert.insert EmptyCollection.emptyCollection S) t
      hst : (Inter.inter s t).Nonempty
      hs : Membership.mem S s
      ⊢ Membership.mem (Insert.insert EmptyCollection.emptyCollection S) (Inter.inte …
    -/
  · cases' ht with ht ht
      /-
        case inr.inl
        α : Type u_1
        S : Set (Set α)
        h_pi : IsPiSystem S
        s t : Set α
        hst : (Inter.inter s t).Nonempty
        hs : Membership.mem S s
        ht : Eq t EmptyCollection.emptyCollection
        ⊢ Membership.mem (Insert.insert EmptyCollection.emptyCollection S) (Inter.inte …
      -/
    · simp [ht]
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        α : Type u_1
        S : Set (Set α)
        h_pi : IsPiSystem S
        s t : Set α
        hst : (Inter.inter s t).Nonempty
        hs : Membership.mem S s
        ht : Membership.mem S t
        ⊢ Membership.mem (Insert.insert EmptyCollection.emptyCollection S) (Inter.inte …
      -/
    · exact Set.mem_insert_of_mem _ (h_pi s hs t ht hst)
      /-
        🎉 no goals
      -/


theorem IsPiSystem.insert_univ {S : Set (Set α)} (h_pi : IsPiSystem S) :
    IsPiSystem (insert Set.univ S) := by
  /-
    α : Type u_1
    S : Set (Set α)
    h_pi : IsPiSystem S
    ⊢ IsPiSystem (Insert.insert Set.univ S)
  -/
  intro s hs t ht hst
  /-
    α : Type u_1
    S : Set (Set α)
    h_pi : IsPiSystem S
    s : Set α
    hs : Membership.mem (Insert.insert Set.univ S) s
    t : Set α
    ht : Membership.mem (Insert.insert Set.univ S) t
    hst : (Inter.inter s t).Nonempty
    ⊢ Membership.mem (Insert.insert Set.univ S) (Inter.inter s t)
  -/
  cases' hs with hs hs
    /-
      case inl
      α : Type u_1
      S : Set (Set α)
      h_pi : IsPiSystem S
      s t : Set α
      ht : Membership.mem (Insert.insert Set.univ S) t
      hst : (Inter.inter s t).Nonempty
      hs : Eq s Set.univ
      ⊢ Membership.mem (Insert.insert Set.univ S) (Inter.inter s t)
    -/
                             /-
                               🎉 no goals
                             -/
  · cases' ht with ht ht <;> simp [hs, ht]
                             /-
                               🎉 no goals
                             -/
    /-
      case inr
      α : Type u_1
      S : Set (Set α)
      h_pi : IsPiSystem S
      s t : Set α
      ht : Membership.mem (Insert.insert Set.univ S) t
      hst : (Inter.inter s t).Nonempty
      hs : Membership.mem S s
      ⊢ Membership.mem (Insert.insert Set.univ S) (Inter.inter s t)
    -/
  · cases' ht with ht ht
      /-
        case inr.inl
        α : Type u_1
        S : Set (Set α)
        h_pi : IsPiSystem S
        s t : Set α
        hst : (Inter.inter s t).Nonempty
        hs : Membership.mem S s
        ht : Eq t Set.univ
        ⊢ Membership.mem (Insert.insert Set.univ S) (Inter.inter s t)
      -/
    · simp [hs, ht]
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        α : Type u_1
        S : Set (Set α)
        h_pi : IsPiSystem S
        s t : Set α
        hst : (Inter.inter s t).Nonempty
        hs : Membership.mem S s
        ht : Membership.mem S t
        ⊢ Membership.mem (Insert.insert Set.univ S) (Inter.inter s t)
      -/
    · exact Set.mem_insert_of_mem _ (h_pi s hs t ht hst)
      /-
        🎉 no goals
      -/


theorem IsPiSystem.comap {α β} {S : Set (Set β)} (h_pi : IsPiSystem S) (f : α → β) :
    IsPiSystem { s : Set α | ∃ t ∈ S, f ⁻¹' t = s } := by
  /-
    α : Type u_3
    β : Type u_4
    S : Set (Set β)
    h_pi : IsPiSystem S
    f : α → β
    ⊢ IsPiSystem (setOf fun s => Exists fun t => And (Membership.mem S t) (Eq (Set …
  -/
  rintro _ ⟨s, hs_mem, rfl⟩ _ ⟨t, ht_mem, rfl⟩ hst
  /-
    case intro.intro.intro.intro
    α : Type u_3
    β : Type u_4
    S : Set (Set β)
    h_pi : IsPiSystem S
    f : α → β
    s : Set β
    hs_mem : Membership.mem S s
    t : Set β
    ht_mem : Membership.mem S t
    hst : (Inter.inter (Set.preimage f s) (Set.preimage f t)).Nonempty
    ⊢ Membership.mem (setOf fun s => Exists fun t => And (Membership.mem S t) (Eq  …
  -/
  rw [← Set.preimage_inter] at hst ⊢
  /-
    case intro.intro.intro.intro
    α : Type u_3
    β : Type u_4
    S : Set (Set β)
    h_pi : IsPiSystem S
    f : α → β
    s : Set β
    hs_mem : Membership.mem S s
    t : Set β
    ht_mem : Membership.mem S t
    hst : (Set.preimage f (Inter.inter s t)).Nonempty
    ⊢ Membership.mem (setOf fun s => Exists fun t => And (Membership.mem S t) (Eq  …
  -/
  exact ⟨s ∩ t, h_pi s hs_mem t ht_mem (nonempty_of_nonempty_preimage hst), rfl⟩
  /-
    🎉 no goals
  -/


theorem isPiSystem_iUnion_of_directed_le {α ι} (p : ι → Set (Set α))
    (hp_pi : ∀ n, IsPiSystem (p n)) (hp_directed : Directed (· ≤ ·) p) :
    IsPiSystem (⋃ n, p n) := by
  /-
    α : Type u_3
    ι : Sort u_4
    p : ι → Set (Set α)
    hp_pi : ∀ (n : ι), IsPiSystem (p n)
    hp_directed : Directed (fun x1 x2 => LE.le x1 x2) p
    ⊢ IsPiSystem (Set.iUnion fun n => p n)
  -/
  intro t1 ht1 t2 ht2 h
  /-
    α : Type u_3
    ι : Sort u_4
    p : ι → Set (Set α)
    hp_pi : ∀ (n : ι), IsPiSystem (p n)
    hp_directed : Directed (fun x1 x2 => LE.le x1 x2) p
    t1 : Set α
    ht1 : Membership.mem (Set.iUnion fun n => p n) t1
    t2 : Set α
    ht2 : Membership.mem (Set.iUnion fun n => p n) t2
    h : (Inter.inter t1 t2).Nonempty
    ⊢ Membership.mem (Set.iUnion fun n => p n) (Inter.inter t1 t2)
  -/
  rw [Set.mem_iUnion] at ht1 ht2 ⊢
  /-
    α : Type u_3
    ι : Sort u_4
    p : ι → Set (Set α)
    hp_pi : ∀ (n : ι), IsPiSystem (p n)
    hp_directed : Directed (fun x1 x2 => LE.le x1 x2) p
    t1 : Set α
    ht1 : Exists fun i => Membership.mem (p i) t1
    t2 : Set α
    ht2 : Exists fun i => Membership.mem (p i) t2
    h : (Inter.inter t1 t2).Nonempty
    ⊢ Exists fun i => Membership.mem (p i) (Inter.inter t1 t2)
  -/
  cases' ht1 with n ht1
  /-
    case intro
    α : Type u_3
    ι : Sort u_4
    p : ι → Set (Set α)
    hp_pi : ∀ (n : ι), IsPiSystem (p n)
    hp_directed : Directed (fun x1 x2 => LE.le x1 x2) p
    t1 t2 : Set α
    ht2 : Exists fun i => Membership.mem (p i) t2
    h : (Inter.inter t1 t2).Nonempty
    n : ι
    ht1 : Membership.mem (p n) t1
    ⊢ Exists fun i => Membership.mem (p i) (Inter.inter t1 t2)
  -/
  cases' ht2 with m ht2
  /-
    case intro.intro
    α : Type u_3
    ι : Sort u_4
    p : ι → Set (Set α)
    hp_pi : ∀ (n : ι), IsPiSystem (p n)
    hp_directed : Directed (fun x1 x2 => LE.le x1 x2) p
    t1 t2 : Set α
    h : (Inter.inter t1 t2).Nonempty
    n : ι
    ht1 : Membership.mem (p n) t1
    m : ι
    ht2 : Membership.mem (p m) t2
    ⊢ Exists fun i => Membership.mem (p i) (Inter.inter t1 t2)
  -/
  obtain ⟨k, hpnk, hpmk⟩ : ∃ k, p n ≤ p k ∧ p m ≤ p k := hp_directed n m
  /-
    case intro.intro.intro.intro
    α : Type u_3
    ι : Sort u_4
    p : ι → Set (Set α)
    hp_pi : ∀ (n : ι), IsPiSystem (p n)
    hp_directed : Directed (fun x1 x2 => LE.le x1 x2) p
    t1 t2 : Set α
    h : (Inter.inter t1 t2).Nonempty
    n : ι
    ht1 : Membership.mem (p n) t1
    m : ι
    ht2 : Membership.mem (p m) t2
    k : ι
    hpnk : LE.le (p n) (p k)
    hpmk : LE.le (p m) (p k)
    ⊢ Exists fun i => Membership.mem (p i) (Inter.inter t1 t2)
  -/
  exact ⟨k, hp_pi k t1 (hpnk ht1) t2 (hpmk ht2) h⟩
  /-
    🎉 no goals
  -/


theorem isPiSystem_iUnion_of_monotone {α ι} [SemilatticeSup ι] (p : ι → Set (Set α))
    (hp_pi : ∀ n, IsPiSystem (p n)) (hp_mono : Monotone p) : IsPiSystem (⋃ n, p n) :=
  isPiSystem_iUnion_of_directed_le p hp_pi (Monotone.directed_le hp_mono)


/-- Rectangles formed by π-systems form a π-system. -/
lemma IsPiSystem.prod {C : Set (Set α)} {D : Set (Set β)} (hC : IsPiSystem C) (hD : IsPiSystem D) :
    IsPiSystem (image2 (· ×ˢ ·) C D) := by
  /-
    α : Type u_1
    β : Type u_2
    C : Set (Set α)
    D : Set (Set β)
    hC : IsPiSystem C
    hD : IsPiSystem D
    ⊢ IsPiSystem (Set.image2 (fun x1 x2 => SProd.sprod x1 x2) C D)
  -/
  rintro _ ⟨s₁, hs₁, t₁, ht₁, rfl⟩ _ ⟨s₂, hs₂, t₂, ht₂, rfl⟩ hst
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    C : Set (Set α)
    D : Set (Set β)
    hC : IsPiSystem C
    hD : IsPiSystem D
    s₁ : Set α
    hs₁ : Membership.mem C s₁
    t₁ : Set β
    ht₁ : Membership.mem D t₁
    s₂ : Set α
    hs₂ : Membership.mem C s₂
    t₂ : Set β
    ht₂ : Membership.mem D t₂
    hst : (Inter.inter ((fun x1 x2 => SProd.sprod x1 x2) s₁ t₁) ((fun x1 x2 => SPr …
    ⊢ Membership.mem (Set.image2 (fun x1 x2 => SProd.sprod x1 x2) C D) (Inter.inte …
  -/
  rw [prod_inter_prod] at hst ⊢; rw [prod_nonempty_iff] at hst
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    C : Set (Set α)
    D : Set (Set β)
    hC : IsPiSystem C
    hD : IsPiSystem D
    s₁ : Set α
    hs₁ : Membership.mem C s₁
    t₁ : Set β
    ht₁ : Membership.mem D t₁
    s₂ : Set α
    hs₂ : Membership.mem C s₂
    t₂ : Set β
    ht₂ : Membership.mem D t₂
    hst : And (Inter.inter s₁ s₂).Nonempty (Inter.inter t₁ t₂).Nonempty
    ⊢ Membership.mem (Set.image2 (fun x1 x2 => SProd.sprod x1 x2) C D) (SProd.spro …
  -/
  exact mem_image2_of_mem (hC _ hs₁ _ hs₂ hst.1) (hD _ ht₁ _ ht₂ hst.2)
  /-
    🎉 no goals
  -/


theorem isPiSystem_image_Iio (s : Set α) : IsPiSystem (Iio '' s) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    ⊢ IsPiSystem (Set.image Set.Iio s)
  -/
  rintro _ ⟨a, ha, rfl⟩ _ ⟨b, hb, rfl⟩ -
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    ⊢ Membership.mem (Set.image Set.Iio s) (Inter.inter (Set.Iio a) (Set.Iio b))
  -/
  exact ⟨a ⊓ b, inf_ind a b ha hb, Iio_inter_Iio.symm⟩
  /-
    🎉 no goals
  -/


theorem isPiSystem_Iio : IsPiSystem (range Iio : Set (Set α)) :=
  @image_univ α _ Iio ▸ isPiSystem_image_Iio univ


theorem isPiSystem_image_Ioi (s : Set α) : IsPiSystem (Ioi '' s) :=
  @isPiSystem_image_Iio αᵒᵈ _ s


theorem isPiSystem_Ioi : IsPiSystem (range Ioi : Set (Set α)) :=
  @image_univ α _ Ioi ▸ isPiSystem_image_Ioi univ


theorem isPiSystem_image_Iic (s : Set α) : IsPiSystem (Iic '' s) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    ⊢ IsPiSystem (Set.image Set.Iic s)
  -/
  rintro _ ⟨a, ha, rfl⟩ _ ⟨b, hb, rfl⟩ -
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    ⊢ Membership.mem (Set.image Set.Iic s) (Inter.inter (Set.Iic a) (Set.Iic b))
  -/
  exact ⟨a ⊓ b, inf_ind a b ha hb, Iic_inter_Iic.symm⟩
  /-
    🎉 no goals
  -/


theorem isPiSystem_Iic : IsPiSystem (range Iic : Set (Set α)) :=
  @image_univ α _ Iic ▸ isPiSystem_image_Iic univ


theorem isPiSystem_image_Ici (s : Set α) : IsPiSystem (Ici '' s) :=
  @isPiSystem_image_Iic αᵒᵈ _ s


theorem isPiSystem_Ici : IsPiSystem (range Ici : Set (Set α)) :=
  @image_univ α _ Ici ▸ isPiSystem_image_Ici univ


theorem isPiSystem_Ixx_mem {Ixx : α → α → Set α} {p : α → α → Prop}
    (Hne : ∀ {a b}, (Ixx a b).Nonempty → p a b)
    (Hi : ∀ {a₁ b₁ a₂ b₂}, Ixx a₁ b₁ ∩ Ixx a₂ b₂ = Ixx (max a₁ a₂) (min b₁ b₂)) (s t : Set α) :
    IsPiSystem { S | ∃ᵉ (l ∈ s) (u ∈ t), p l u ∧ Ixx l u = S } := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    Ixx : α → α → Set α
    p : α → α → Prop
    Hne : ∀ {a b : α}, (Ixx a b).Nonempty → p a b
    Hi : ∀ {a₁ b₁ a₂ b₂ : α}, Eq (Inter.inter (Ixx a₁ b₁) (Ixx a₂ b₂)) (Ixx (Max.m …
    s t : Set α
    ⊢ IsPiSystem (setOf fun S => Exists fun l => And (Membership.mem s l) (Exists  …
  -/
  rintro _ ⟨l₁, hls₁, u₁, hut₁, _, rfl⟩ _ ⟨l₂, hls₂, u₂, hut₂, _, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrder α
    Ixx : α → α → Set α
    p : α → α → Prop
    Hne : ∀ {a b : α}, (Ixx a b).Nonempty → p a b
    Hi : ∀ {a₁ b₁ a₂ b₂ : α}, Eq (Inter.inter (Ixx a₁ b₁) (Ixx a₂ b₂)) (Ixx (Max.m …
    s t : Set α
    l₁ : α
    hls₁ : Membership.mem s l₁
    u₁ : α
    hut₁ : Membership.mem t u₁
    left✝¹ : p l₁ u₁
    l₂ : α
    hls₂ : Membership.mem s l₂
    u₂ : α
    hut₂ : Membership.mem t u₂
    left✝ : p l₂ u₂
    ⊢ (Inter.inter (Ixx l₁ u₁) (Ixx l₂ u₂)).Nonempty → Membership.mem (setOf fun S …
  -/
  simp only [Hi]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrder α
    Ixx : α → α → Set α
    p : α → α → Prop
    Hne : ∀ {a b : α}, (Ixx a b).Nonempty → p a b
    Hi : ∀ {a₁ b₁ a₂ b₂ : α}, Eq (Inter.inter (Ixx a₁ b₁) (Ixx a₂ b₂)) (Ixx (Max.m …
    s t : Set α
    l₁ : α
    hls₁ : Membership.mem s l₁
    u₁ : α
    hut₁ : Membership.mem t u₁
    left✝¹ : p l₁ u₁
    l₂ : α
    hls₂ : Membership.mem s l₂
    u₂ : α
    hut₂ : Membership.mem t u₂
    left✝ : p l₂ u₂
    ⊢ (Ixx (Max.max l₁ l₂) (Min.min u₁ u₂)).Nonempty → Membership.mem (setOf fun S …
  -/
  exact fun H => ⟨l₁ ⊔ l₂, sup_ind l₁ l₂ hls₁ hls₂, u₁ ⊓ u₂, inf_ind u₁ u₂ hut₁ hut₂, Hne H, rfl⟩
  /-
    🎉 no goals
  -/


theorem isPiSystem_Ixx {Ixx : α → α → Set α} {p : α → α → Prop}
    (Hne : ∀ {a b}, (Ixx a b).Nonempty → p a b)
    (Hi : ∀ {a₁ b₁ a₂ b₂}, Ixx a₁ b₁ ∩ Ixx a₂ b₂ = Ixx (max a₁ a₂) (min b₁ b₂)) (f : ι → α)
    (g : ι' → α) : @IsPiSystem α { S | ∃ i j, p (f i) (g j) ∧ Ixx (f i) (g j) = S } := by
  /-
    α : Type u_1
    ι : Sort u_3
    ι' : Sort u_4
    inst✝ : LinearOrder α
    Ixx : α → α → Set α
    p : α → α → Prop
    Hne : ∀ {a b : α}, (Ixx a b).Nonempty → p a b
    Hi : ∀ {a₁ b₁ a₂ b₂ : α}, Eq (Inter.inter (Ixx a₁ b₁) (Ixx a₂ b₂)) (Ixx (Max.m …
    f : ι → α
    g : ι' → α
    ⊢ IsPiSystem (setOf fun S => Exists fun i => Exists fun j => And (p (f i) (g j …
  -/
  simpa only [exists_range_iff] using isPiSystem_Ixx_mem (@Hne) (@Hi) (range f) (range g)
  /-
    🎉 no goals
  -/


theorem isPiSystem_Ioo_mem (s t : Set α) :
    IsPiSystem { S | ∃ᵉ (l ∈ s) (u ∈ t), l < u ∧ Ioo l u = S } :=
  isPiSystem_Ixx_mem (Ixx := Ioo) (fun ⟨_, hax, hxb⟩ => hax.trans hxb) Ioo_inter_Ioo s t


theorem isPiSystem_Ioo (f : ι → α) (g : ι' → α) :
    @IsPiSystem α { S | ∃ l u, f l < g u ∧ Ioo (f l) (g u) = S } :=
  isPiSystem_Ixx (Ixx := Ioo) (fun ⟨_, hax, hxb⟩ => hax.trans hxb) Ioo_inter_Ioo f g


theorem isPiSystem_Ioc_mem (s t : Set α) :
    IsPiSystem { S | ∃ᵉ (l ∈ s) (u ∈ t), l < u ∧ Ioc l u = S } :=
  isPiSystem_Ixx_mem (Ixx := Ioc) (fun ⟨_, hax, hxb⟩ => hax.trans_le hxb) Ioc_inter_Ioc s t


theorem isPiSystem_Ioc (f : ι → α) (g : ι' → α) :
    @IsPiSystem α { S | ∃ i j, f i < g j ∧ Ioc (f i) (g j) = S } :=
  isPiSystem_Ixx (Ixx := Ioc) (fun ⟨_, hax, hxb⟩ => hax.trans_le hxb) Ioc_inter_Ioc f g


theorem isPiSystem_Ico_mem (s t : Set α) :
    IsPiSystem { S | ∃ᵉ (l ∈ s) (u ∈ t), l < u ∧ Ico l u = S } :=
  isPiSystem_Ixx_mem (Ixx := Ico) (fun ⟨_, hax, hxb⟩ => hax.trans_lt hxb) Ico_inter_Ico s t


theorem isPiSystem_Ico (f : ι → α) (g : ι' → α) :
    @IsPiSystem α { S | ∃ i j, f i < g j ∧ Ico (f i) (g j) = S } :=
  isPiSystem_Ixx (Ixx := Ico) (fun ⟨_, hax, hxb⟩ => hax.trans_lt hxb) Ico_inter_Ico f g


theorem isPiSystem_Icc_mem (s t : Set α) :
    IsPiSystem { S | ∃ᵉ (l ∈ s) (u ∈ t), l ≤ u ∧ Icc l u = S } :=
                                                     /-
                                                       α : Type u_1
                                                       inst✝ : LinearOrder α
                                                       s t : Set α
                                                       ⊢ ∀ {a₁ b₁ a₂ b₂ : α}, Eq (Inter.inter (Set.Icc a₁ b₁) (Set.Icc a₂ b₂)) (Set.I …
                                                     -/
  isPiSystem_Ixx_mem (Ixx := Icc) nonempty_Icc.1 (by exact Icc_inter_Icc) s t
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem isPiSystem_Icc (f : ι → α) (g : ι' → α) :
    @IsPiSystem α { S | ∃ i j, f i ≤ g j ∧ Icc (f i) (g j) = S } :=
                                                 /-
                                                   α : Type u_1
                                                   ι : Sort u_3
                                                   ι' : Sort u_4
                                                   inst✝ : LinearOrder α
                                                   f : ι → α
                                                   g : ι' → α
                                                   ⊢ ∀ {a₁ b₁ a₂ b₂ : α}, Eq (Inter.inter (Set.Icc a₁ b₁) (Set.Icc a₂ b₂)) (Set.I …
                                                 -/
  isPiSystem_Ixx (Ixx := Icc) nonempty_Icc.1 (by exact Icc_inter_Icc) f g
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- Given a collection `S` of subsets of `α`, then `generatePiSystem S` is the smallest
π-system containing `S`. -/
inductive generatePiSystem (S : Set (Set α)) : Set (Set α)
  | base {s : Set α} (h_s : s ∈ S) : generatePiSystem S s
  | inter {s t : Set α} (h_s : generatePiSystem S s) (h_t : generatePiSystem S t)
    (h_nonempty : (s ∩ t).Nonempty) : generatePiSystem S (s ∩ t)


theorem isPiSystem_generatePiSystem (S : Set (Set α)) : IsPiSystem (generatePiSystem S) :=
  fun _ h_s _ h_t h_nonempty => generatePiSystem.inter h_s h_t h_nonempty


theorem subset_generatePiSystem_self (S : Set (Set α)) : S ⊆ generatePiSystem S := fun _ =>
  generatePiSystem.base


theorem generatePiSystem_subset_self {S : Set (Set α)} (h_S : IsPiSystem S) :
    generatePiSystem S ⊆ S := fun x h => by
  induction h with
  | base h_s => exact h_s
  | inter _ _ h_nonempty h_s h_u => exact h_S _ h_s _ h_u h_nonempty


theorem generatePiSystem_eq {S : Set (Set α)} (h_pi : IsPiSystem S) : generatePiSystem S = S :=
  Set.Subset.antisymm (generatePiSystem_subset_self h_pi) (subset_generatePiSystem_self S)


theorem generatePiSystem_mono {S T : Set (Set α)} (hST : S ⊆ T) :
    generatePiSystem S ⊆ generatePiSystem T := fun t ht => by
  induction ht with
  | base h_s => exact generatePiSystem.base (Set.mem_of_subset_of_mem hST h_s)
  | inter _ _ h_nonempty h_s h_u => exact isPiSystem_generatePiSystem T _ h_s _ h_u h_nonempty


theorem generatePiSystem_measurableSet [M : MeasurableSpace α] {S : Set (Set α)}
    (h_meas_S : ∀ s ∈ S, MeasurableSet s) (t : Set α) (h_in_pi : t ∈ generatePiSystem S) :
    MeasurableSet t := by
  induction h_in_pi with
  | base h_s => apply h_meas_S _ h_s
  | inter _ _ _ h_s h_u => apply MeasurableSet.inter h_s h_u


theorem generateFrom_measurableSet_of_generatePiSystem {g : Set (Set α)} (t : Set α)
    (ht : t ∈ generatePiSystem g) : MeasurableSet[generateFrom g] t :=
  @generatePiSystem_measurableSet α (generateFrom g) g
    (fun _ h_s_in_g => measurableSet_generateFrom h_s_in_g) t ht


theorem generateFrom_generatePiSystem_eq {g : Set (Set α)} :
    generateFrom (generatePiSystem g) = generateFrom g := by
  /-
    α : Type u_1
    g : Set (Set α)
    ⊢ Eq (MeasurableSpace.generateFrom (generatePiSystem g)) (MeasurableSpace.gene …
  -/
  apply le_antisymm <;> apply generateFrom_le
    /-
      case a.h
      α : Type u_1
      g : Set (Set α)
      ⊢ ∀ (t : Set α), Membership.mem (generatePiSystem g) t → MeasurableSet t
    -/
  · exact fun t h_t => generateFrom_measurableSet_of_generatePiSystem t h_t
    /-
      🎉 no goals
    -/
    /-
      case a.h
      α : Type u_1
      g : Set (Set α)
      ⊢ ∀ (t : Set α), Membership.mem g t → MeasurableSet t
    -/
  · exact fun t h_t => measurableSet_generateFrom (generatePiSystem.base h_t)
    /-
      🎉 no goals
    -/


/-- Every element of the π-system generated by the union of a family of π-systems
is a finite intersection of elements from the π-systems.
For an indexed union version, see `mem_generatePiSystem_iUnion_elim'`. -/
theorem mem_generatePiSystem_iUnion_elim {α β} {g : β → Set (Set α)} (h_pi : ∀ b, IsPiSystem (g b))
    (t : Set α) (h_t : t ∈ generatePiSystem (⋃ b, g b)) :
    ∃ (T : Finset β) (f : β → Set α), (t = ⋂ b ∈ T, f b) ∧ ∀ b ∈ T, f b ∈ g b := by
  classical
  induction' h_t with s h_s s t' h_gen_s h_gen_t' h_nonempty h_s h_t'
  · rcases h_s with ⟨t', ⟨⟨b, rfl⟩, h_s_in_t'⟩⟩
    refine ⟨{b}, fun _ => s, ?_⟩
    simpa using h_s_in_t'
  · rcases h_t' with ⟨T_t', ⟨f_t', ⟨rfl, h_t'⟩⟩⟩
    rcases h_s with ⟨T_s, ⟨f_s, ⟨rfl, h_s⟩⟩⟩
    use T_s ∪ T_t', fun b : β =>
      if b ∈ T_s then if b ∈ T_t' then f_s b ∩ f_t' b else f_s b
      else if b ∈ T_t' then f_t' b else (∅ : Set α)
    constructor
    · ext a
      simp_rw [Set.mem_inter_iff, Set.mem_iInter, Finset.mem_union, or_imp]
      rw [← forall_and]
      constructor <;> intro h1 b <;> by_cases hbs : b ∈ T_s <;> by_cases hbt : b ∈ T_t' <;>
          specialize h1 b <;>
        simp only [hbs, hbt, if_true, if_false, true_imp_iff, and_self_iff, false_imp_iff] at h1 ⊢
      all_goals exact h1
    intro b h_b
    split_ifs with hbs hbt hbt
    · refine h_pi b (f_s b) (h_s b hbs) (f_t' b) (h_t' b hbt) (Set.Nonempty.mono ?_ h_nonempty)
      exact Set.inter_subset_inter (Set.biInter_subset_of_mem hbs) (Set.biInter_subset_of_mem hbt)
    · exact h_s b hbs
    · exact h_t' b hbt
    · rw [Finset.mem_union] at h_b
      apply False.elim (h_b.elim hbs hbt)


/-- Every element of the π-system generated by an indexed union of a family of π-systems
is a finite intersection of elements from the π-systems.
For a total union version, see `mem_generatePiSystem_iUnion_elim`. -/
theorem mem_generatePiSystem_iUnion_elim' {α β} {g : β → Set (Set α)} {s : Set β}
    (h_pi : ∀ b ∈ s, IsPiSystem (g b)) (t : Set α) (h_t : t ∈ generatePiSystem (⋃ b ∈ s, g b)) :
    ∃ (T : Finset β) (f : β → Set α), ↑T ⊆ s ∧ (t = ⋂ b ∈ T, f b) ∧ ∀ b ∈ T, f b ∈ g b := by
  classical
  have : t ∈ generatePiSystem (⋃ b : Subtype s, (g ∘ Subtype.val) b) := by
    suffices h1 : ⋃ b : Subtype s, (g ∘ Subtype.val) b = ⋃ b ∈ s, g b by rwa [h1]
    ext x
    simp only [exists_prop, Set.mem_iUnion, Function.comp_apply, Subtype.exists, Subtype.coe_mk]
    rfl
  rcases @mem_generatePiSystem_iUnion_elim α (Subtype s) (g ∘ Subtype.val)
      (fun b => h_pi b.val b.property) t this with
    ⟨T, ⟨f, ⟨rfl, h_t'⟩⟩⟩
  refine
    ⟨T.image (fun x : s => (x : β)),
      Function.extend (fun x : s => (x : β)) f fun _ : β => (∅ : Set α), by simp, ?_, ?_⟩
  · ext a
    constructor <;>
      · simp (config := { proj := false }) only
          [Set.mem_iInter, Subtype.forall, Finset.set_biInter_finset_image]
        intro h1 b h_b h_b_in_T
        have h2 := h1 b h_b h_b_in_T
        revert h2
        rw [Subtype.val_injective.extend_apply]
        apply id
  · intros b h_b
    simp_rw [Finset.mem_image, Subtype.exists, exists_and_right, exists_eq_right]
      at h_b
    cases' h_b with h_b_w h_b_h
    have h_b_alt : b = (Subtype.mk b h_b_w).val := rfl
    rw [h_b_alt, Subtype.val_injective.extend_apply]
    apply h_t'
    apply h_b_h


/-- From a set of indices `S : Set ι` and a family of sets of sets `π : ι → Set (Set α)`,
define the set of sets that can be written as `⋂ x ∈ t, f x` for some finset `t ⊆ S` and sets
`f x ∈ π x`. If `π` is a family of π-systems, then it is a π-system. -/
def piiUnionInter (π : ι → Set (Set α)) (S : Set ι) : Set (Set α) :=
  { s : Set α |
    ∃ (t : Finset ι) (_ : ↑t ⊆ S) (f : ι → Set α) (_ : ∀ x, x ∈ t → f x ∈ π x), s = ⋂ x ∈ t, f x }


theorem piiUnionInter_singleton (π : ι → Set (Set α)) (i : ι) :
    piiUnionInter π {i} = π i ∪ {univ} := by
  /-
    α : Type u_3
    ι : Type u_4
    π : ι → Set (Set α)
    i : ι
    ⊢ Eq (piiUnionInter π (Singleton.singleton i)) (Union.union (π i) (Singleton.s …
  -/
  ext1 s
  /-
    case h
    α : Type u_3
    ι : Type u_4
    π : ι → Set (Set α)
    i : ι
    s : Set α
    ⊢ Iff (Membership.mem (piiUnionInter π (Singleton.singleton i)) s) (Membership …
  -/
  simp only [piiUnionInter, exists_prop, mem_union]
  /-
    case h
    α : Type u_3
    ι : Type u_4
    π : ι → Set (Set α)
    i : ι
    s : Set α
    ⊢ Iff (Membership.mem (setOf fun s => Exists fun t => And (HasSubset.Subset (↑ …
  -/
  refine ⟨?_, fun h => ?_⟩
    /-
      case h.refine_1
      α : Type u_3
      ι : Type u_4
      π : ι → Set (Set α)
      i : ι
      s : Set α
      ⊢ Membership.mem (setOf fun s => Exists fun t => And (HasSubset.Subset (↑t) (S …
    -/
  · rintro ⟨t, hti, f, hfπ, rfl⟩
    /-
      case h.refine_1.intro.intro.intro.intro
      α : Type u_3
      ι : Type u_4
      π : ι → Set (Set α)
      i : ι
      t : Finset ι
      hti : HasSubset.Subset (↑t) (Singleton.singleton i)
      f : ι → Set α
      hfπ : ∀ (x : ι), Membership.mem t x → Membership.mem (π x) (f x)
      ⊢ Or (Membership.mem (π i) (Set.iInter fun x => Set.iInter fun h => f x)) (Mem …
    -/
    simp only [subset_singleton_iff, Finset.mem_coe] at hti
    /-
      case h.refine_1.intro.intro.intro.intro
      α : Type u_3
      ι : Type u_4
      π : ι → Set (Set α)
      i : ι
      t : Finset ι
      f : ι → Set α
      hfπ : ∀ (x : ι), Membership.mem t x → Membership.mem (π x) (f x)
      hti : ∀ (y : ι), Membership.mem t y → Eq y i
      ⊢ Or (Membership.mem (π i) (Set.iInter fun x => Set.iInter fun h => f x)) (Mem …
    -/
    by_cases hi : i ∈ t
    · have ht_eq_i : t = {i} := by
        ext1 x
        rw [Finset.mem_singleton]
        exact ⟨fun h => hti x h, fun h => h.symm ▸ hi⟩
      /-
        case pos
        α : Type u_3
        ι : Type u_4
        π : ι → Set (Set α)
        i : ι
        t : Finset ι
        f : ι → Set α
        hfπ : ∀ (x : ι), Membership.mem t x → Membership.mem (π x) (f x)
        hti : ∀ (y : ι), Membership.mem t y → Eq y i
        hi : Membership.mem t i
        ht_eq_i : Eq t (Singleton.singleton i)
        ⊢ Or (Membership.mem (π i) (Set.iInter fun x => Set.iInter fun h => f x)) (Mem …
      -/
      simp only [ht_eq_i, Finset.mem_singleton, iInter_iInter_eq_left]
      /-
        case pos
        α : Type u_3
        ι : Type u_4
        π : ι → Set (Set α)
        i : ι
        t : Finset ι
        f : ι → Set α
        hfπ : ∀ (x : ι), Membership.mem t x → Membership.mem (π x) (f x)
        hti : ∀ (y : ι), Membership.mem t y → Eq y i
        hi : Membership.mem t i
        ht_eq_i : Eq t (Singleton.singleton i)
        ⊢ Or (Membership.mem (π i) (f i)) (Membership.mem (Singleton.singleton Set.uni …
      -/
      exact Or.inl (hfπ i hi)
      /-
        🎉 no goals
      -/
    · have ht_empty : t = ∅ := by
        ext1 x
        simp only [Finset.not_mem_empty, iff_false]
        exact fun hx => hi (hti x hx ▸ hx)
      -- Porting note: `Finset.not_mem_empty` required
      /-
        case neg
        α : Type u_3
        ι : Type u_4
        π : ι → Set (Set α)
        i : ι
        t : Finset ι
        f : ι → Set α
        hfπ : ∀ (x : ι), Membership.mem t x → Membership.mem (π x) (f x)
        hti : ∀ (y : ι), Membership.mem t y → Eq y i
        hi : Not (Membership.mem t i)
        ht_empty : Eq t EmptyCollection.emptyCollection
        ⊢ Or (Membership.mem (π i) (Set.iInter fun x => Set.iInter fun h => f x)) (Mem …
      -/
      simp [ht_empty, Finset.not_mem_empty, iInter_false, iInter_univ, Set.mem_singleton univ]
      /-
        🎉 no goals
      -/
    /-
      case h.refine_2
      α : Type u_3
      ι : Type u_4
      π : ι → Set (Set α)
      i : ι
      s : Set α
      h : Or (Membership.mem (π i) s) (Membership.mem (Singleton.singleton Set.univ) …
      ⊢ Membership.mem (setOf fun s => Exists fun t => And (HasSubset.Subset (↑t) (S …
    -/
  · cases' h with hs hs
      /-
        case h.refine_2.inl
        α : Type u_3
        ι : Type u_4
        π : ι → Set (Set α)
        i : ι
        s : Set α
        hs : Membership.mem (π i) s
        ⊢ Membership.mem (setOf fun s => Exists fun t => And (HasSubset.Subset (↑t) (S …
      -/
    · refine ⟨{i}, ?_, fun _ => s, ⟨fun x hx => ?_, ?_⟩⟩
        /-
          case h.refine_2.inl.refine_1
          α : Type u_3
          ι : Type u_4
          π : ι → Set (Set α)
          i : ι
          s : Set α
          hs : Membership.mem (π i) s
          ⊢ HasSubset.Subset (↑(Singleton.singleton i)) (Singleton.singleton i)
        -/
      · rw [Finset.coe_singleton]
        /-
          🎉 no goals
        -/
        /-
          case h.refine_2.inl.refine_2
          α : Type u_3
          ι : Type u_4
          π : ι → Set (Set α)
          i : ι
          s : Set α
          hs : Membership.mem (π i) s
          x : ι
          hx : Membership.mem (Singleton.singleton i) x
          ⊢ Membership.mem (π x) ((fun x => s) x)
        -/
      · rw [Finset.mem_singleton] at hx
        /-
          case h.refine_2.inl.refine_2
          α : Type u_3
          ι : Type u_4
          π : ι → Set (Set α)
          i : ι
          s : Set α
          hs : Membership.mem (π i) s
          x : ι
          hx : Eq x i
          ⊢ Membership.mem (π x) ((fun x => s) x)
        -/
        rwa [hx]
        /-
          🎉 no goals
        -/
        /-
          case h.refine_2.inl.refine_3
          α : Type u_3
          ι : Type u_4
          π : ι → Set (Set α)
          i : ι
          s : Set α
          hs : Membership.mem (π i) s
          ⊢ Eq s (Set.iInter fun x => Set.iInter fun h => (fun x => s) x)
        -/
      · simp only [Finset.mem_singleton, iInter_iInter_eq_left]
        /-
          🎉 no goals
        -/
      /-
        case h.refine_2.inr
        α : Type u_3
        ι : Type u_4
        π : ι → Set (Set α)
        i : ι
        s : Set α
        hs : Membership.mem (Singleton.singleton Set.univ) s
        ⊢ Membership.mem (setOf fun s => Exists fun t => And (HasSubset.Subset (↑t) (S …
      -/
    · refine ⟨∅, ?_⟩
      simpa only [Finset.coe_empty, subset_singleton_iff, mem_empty_iff_false, IsEmpty.forall_iff,
        imp_true_iff, Finset.not_mem_empty, iInter_false, iInter_univ, true_and,
        exists_const] using hs


theorem piiUnionInter_singleton_left (s : ι → Set α) (S : Set ι) :
    piiUnionInter (fun i => ({s i} : Set (Set α))) S =
      { s' : Set α | ∃ (t : Finset ι) (_ : ↑t ⊆ S), s' = ⋂ i ∈ t, s i } := by
  /-
    α : Type u_3
    ι : Type u_4
    s : ι → Set α
    S : Set ι
    ⊢ Eq (piiUnionInter (fun i => Singleton.singleton (s i)) S) (setOf fun s' => E …
  -/
  ext1 s'
  /-
    case h
    α : Type u_3
    ι : Type u_4
    s : ι → Set α
    S : Set ι
    s' : Set α
    ⊢ Iff (Membership.mem (piiUnionInter (fun i => Singleton.singleton (s i)) S) s …
  -/
  simp_rw [piiUnionInter, Set.mem_singleton_iff, exists_prop, Set.mem_setOf_eq]
  /-
    case h
    α : Type u_3
    ι : Type u_4
    s : ι → Set α
    S : Set ι
    s' : Set α
    ⊢ Iff (Exists fun t => And (HasSubset.Subset (↑t) S) (Exists fun f => And (∀ ( …
  -/
  refine ⟨fun h => ?_, fun ⟨t, htS, h_eq⟩ => ⟨t, htS, s, fun _ _ => rfl, h_eq⟩⟩
  /-
    case h
    α : Type u_3
    ι : Type u_4
    s : ι → Set α
    S : Set ι
    s' : Set α
    h : Exists fun t => And (HasSubset.Subset (↑t) S) (Exists fun f => And (∀ (x : …
    ⊢ Exists fun t => And (HasSubset.Subset (↑t) S) (Eq s' (Set.iInter fun i => Se …
  -/
  obtain ⟨t, htS, f, hft_eq, rfl⟩ := h
  /-
    case h.intro.intro.intro.intro
    α : Type u_3
    ι : Type u_4
    s : ι → Set α
    S : Set ι
    t : Finset ι
    htS : HasSubset.Subset (↑t) S
    f : ι → Set α
    hft_eq : ∀ (x : ι), Membership.mem t x → Eq (f x) (s x)
    ⊢ Exists fun t_1 => And (HasSubset.Subset (↑t_1) S) (Eq (Set.iInter fun x => S …
  -/
  refine ⟨t, htS, ?_⟩
  /-
    case h.intro.intro.intro.intro
    α : Type u_3
    ι : Type u_4
    s : ι → Set α
    S : Set ι
    t : Finset ι
    htS : HasSubset.Subset (↑t) S
    f : ι → Set α
    hft_eq : ∀ (x : ι), Membership.mem t x → Eq (f x) (s x)
    ⊢ Eq (Set.iInter fun x => Set.iInter fun h => f x) (Set.iInter fun i => Set.iI …
  -/
  congr! 3
  /-
    case h.intro.intro.intro.intro.h.e'_3.h.f
    α : Type u_3
    ι : Type u_4
    s : ι → Set α
    S : Set ι
    t : Finset ι
    htS : HasSubset.Subset (↑t) S
    f : ι → Set α
    hft_eq : ∀ (x : ι), Membership.mem t x → Eq (f x) (s x)
    x✝¹ : ι
    x✝ : Membership.mem t x✝¹
    ⊢ Eq (f x✝¹) (s x✝¹)
  -/
  apply hft_eq
  /-
    case h.intro.intro.intro.intro.h.e'_3.h.f.a
    α : Type u_3
    ι : Type u_4
    s : ι → Set α
    S : Set ι
    t : Finset ι
    htS : HasSubset.Subset (↑t) S
    f : ι → Set α
    hft_eq : ∀ (x : ι), Membership.mem t x → Eq (f x) (s x)
    x✝¹ : ι
    x✝ : Membership.mem t x✝¹
    ⊢ Membership.mem t x✝¹
  -/
  assumption
  /-
    🎉 no goals
  -/


theorem generateFrom_piiUnionInter_singleton_left (s : ι → Set α) (S : Set ι) :
    generateFrom (piiUnionInter (fun k => {s k}) S) = generateFrom { t | ∃ k ∈ S, s k = t } := by
  /-
    α : Type u_3
    ι : Type u_4
    s : ι → Set α
    S : Set ι
    ⊢ Eq (MeasurableSpace.generateFrom (piiUnionInter (fun k => Singleton.singleto …
  -/
  refine le_antisymm (generateFrom_le ?_) (generateFrom_mono ?_)
    /-
      case refine_1
      α : Type u_3
      ι : Type u_4
      s : ι → Set α
      S : Set ι
      ⊢ ∀ (t : Set α), Membership.mem (piiUnionInter (fun k => Singleton.singleton ( …
    -/
  · rintro _ ⟨I, hI, f, hf, rfl⟩
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_3
      ι : Type u_4
      s : ι → Set α
      S : Set ι
      I : Finset ι
      hI : HasSubset.Subset (↑I) S
      f : ι → Set α
      hf : ∀ (x : ι), Membership.mem I x → Membership.mem ((fun k => Singleton.singl …
      ⊢ MeasurableSet (Set.iInter fun x => Set.iInter fun h => f x)
    -/
    refine Finset.measurableSet_biInter _ fun m hm => measurableSet_generateFrom ?_
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_3
      ι : Type u_4
      s : ι → Set α
      S : Set ι
      I : Finset ι
      hI : HasSubset.Subset (↑I) S
      f : ι → Set α
      hf : ∀ (x : ι), Membership.mem I x → Membership.mem ((fun k => Singleton.singl …
      m : ι
      hm : Membership.mem I m
      ⊢ Membership.mem (setOf fun t => Exists fun k => And (Membership.mem S k) (Eq  …
    -/
    exact ⟨m, hI hm, (hf m hm).symm⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_3
      ι : Type u_4
      s : ι → Set α
      S : Set ι
      ⊢ HasSubset.Subset (setOf fun t => Exists fun k => And (Membership.mem S k) (E …
    -/
  · rintro _ ⟨k, hk, rfl⟩
    /-
      case refine_2.intro.intro
      α : Type u_3
      ι : Type u_4
      s : ι → Set α
      S : Set ι
      k : ι
      hk : Membership.mem S k
      ⊢ Membership.mem (piiUnionInter (fun k => Singleton.singleton (s k)) S) (s k)
    -/
    refine ⟨{k}, fun m hm => ?_, s, fun i _ => ?_, ?_⟩
      /-
        case refine_2.intro.intro.refine_1
        α : Type u_3
        ι : Type u_4
        s : ι → Set α
        S : Set ι
        k : ι
        hk : Membership.mem S k
        m : ι
        hm : Membership.mem (↑(Singleton.singleton k)) m
        ⊢ Membership.mem S m
      -/
    · rw [Finset.mem_coe, Finset.mem_singleton] at hm
      /-
        case refine_2.intro.intro.refine_1
        α : Type u_3
        ι : Type u_4
        s : ι → Set α
        S : Set ι
        k : ι
        hk : Membership.mem S k
        m : ι
        hm : Eq m k
        ⊢ Membership.mem S m
      -/
      rwa [hm]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.refine_2
        α : Type u_3
        ι : Type u_4
        s : ι → Set α
        S : Set ι
        k : ι
        hk : Membership.mem S k
        i : ι
        x✝ : Membership.mem (Singleton.singleton k) i
        ⊢ Membership.mem ((fun k => Singleton.singleton (s k)) i) (s i)
      -/
    · exact Set.mem_singleton _
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.refine_3
        α : Type u_3
        ι : Type u_4
        s : ι → Set α
        S : Set ι
        k : ι
        hk : Membership.mem S k
        ⊢ Eq (s k) (Set.iInter fun x => Set.iInter fun h => s x)
      -/
    · simp only [Finset.mem_singleton, Set.iInter_iInter_eq_left]
      /-
        🎉 no goals
      -/


/-- If `π` is a family of π-systems, then `piiUnionInter π S` is a π-system. -/
theorem isPiSystem_piiUnionInter (π : ι → Set (Set α)) (hpi : ∀ x, IsPiSystem (π x)) (S : Set ι) :
    IsPiSystem (piiUnionInter π S) := by
  classical
  rintro t1 ⟨p1, hp1S, f1, hf1m, ht1_eq⟩ t2 ⟨p2, hp2S, f2, hf2m, ht2_eq⟩ h_nonempty
  simp_rw [piiUnionInter, Set.mem_setOf_eq]
  let g n := ite (n ∈ p1) (f1 n) Set.univ ∩ ite (n ∈ p2) (f2 n) Set.univ
  have hp_union_ss : ↑(p1 ∪ p2) ⊆ S := by
    simp only [hp1S, hp2S, Finset.coe_union, union_subset_iff, and_self_iff]
  use p1 ∪ p2, hp_union_ss, g
  have h_inter_eq : t1 ∩ t2 = ⋂ i ∈ p1 ∪ p2, g i := by
    rw [ht1_eq, ht2_eq]
    simp_rw [← Set.inf_eq_inter]
    ext1 x
    simp only [g, inf_eq_inter, mem_inter_iff, mem_iInter, Finset.mem_union]
    refine ⟨fun h i _ => ?_, fun h => ⟨fun i hi1 => ?_, fun i hi2 => ?_⟩⟩
    · split_ifs with h_1 h_2 h_2
      exacts [⟨h.1 i h_1, h.2 i h_2⟩, ⟨h.1 i h_1, Set.mem_univ _⟩, ⟨Set.mem_univ _, h.2 i h_2⟩,
        ⟨Set.mem_univ _, Set.mem_univ _⟩]
    · specialize h i (Or.inl hi1)
      rw [if_pos hi1] at h
      exact h.1
    · specialize h i (Or.inr hi2)
      rw [if_pos hi2] at h
      exact h.2
  refine ⟨fun n hn => ?_, h_inter_eq⟩
  simp only [g]
  split_ifs with hn1 hn2 h
  · refine hpi n (f1 n) (hf1m n hn1) (f2 n) (hf2m n hn2) (Set.nonempty_iff_ne_empty.2 fun h => ?_)
    rw [h_inter_eq] at h_nonempty
    suffices h_empty : ⋂ i ∈ p1 ∪ p2, g i = ∅ from
      (Set.not_nonempty_iff_eq_empty.mpr h_empty) h_nonempty
    refine le_antisymm (Set.iInter_subset_of_subset n ?_) (Set.empty_subset _)
    refine Set.iInter_subset_of_subset hn ?_
    simp_rw [g, if_pos hn1, if_pos hn2]
    exact h.subset
  · simp [hf1m n hn1]
  · simp [hf2m n h]
  · exact absurd hn (by simp [hn1, h])


theorem piiUnionInter_mono_left {π π' : ι → Set (Set α)} (h_le : ∀ i, π i ⊆ π' i) (S : Set ι) :
    piiUnionInter π S ⊆ piiUnionInter π' S := fun _ ⟨t, ht_mem, ft, hft_mem_pi, h_eq⟩ =>
  ⟨t, ht_mem, ft, fun x hxt => h_le x (hft_mem_pi x hxt), h_eq⟩


theorem piiUnionInter_mono_right {π : ι → Set (Set α)} {S T : Set ι} (hST : S ⊆ T) :
    piiUnionInter π S ⊆ piiUnionInter π T := fun _ ⟨t, ht_mem, ft, hft_mem_pi, h_eq⟩ =>
  ⟨t, ht_mem.trans hST, ft, hft_mem_pi, h_eq⟩


theorem generateFrom_piiUnionInter_le {m : MeasurableSpace α} (π : ι → Set (Set α))
    (h : ∀ n, generateFrom (π n) ≤ m) (S : Set ι) : generateFrom (piiUnionInter π S) ≤ m := by
  /-
    α : Type u_3
    ι : Type u_4
    m : MeasurableSpace α
    π : ι → Set (Set α)
    h : ∀ (n : ι), LE.le (MeasurableSpace.generateFrom (π n)) m
    S : Set ι
    ⊢ LE.le (MeasurableSpace.generateFrom (piiUnionInter π S)) m
  -/
  refine generateFrom_le ?_
  /-
    α : Type u_3
    ι : Type u_4
    m : MeasurableSpace α
    π : ι → Set (Set α)
    h : ∀ (n : ι), LE.le (MeasurableSpace.generateFrom (π n)) m
    S : Set ι
    ⊢ ∀ (t : Set α), Membership.mem (piiUnionInter π S) t → MeasurableSet t
  -/
  rintro t ⟨ht_p, _, ft, hft_mem_pi, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_3
    ι : Type u_4
    m : MeasurableSpace α
    π : ι → Set (Set α)
    h : ∀ (n : ι), LE.le (MeasurableSpace.generateFrom (π n)) m
    S : Set ι
    ht_p : Finset ι
    w✝ : HasSubset.Subset (↑ht_p) S
    ft : ι → Set α
    hft_mem_pi : ∀ (x : ι), Membership.mem ht_p x → Membership.mem (π x) (ft x)
    ⊢ MeasurableSet (Set.iInter fun x => Set.iInter fun h => ft x)
  -/
  refine Finset.measurableSet_biInter _ fun x hx_mem => (h x) _ ?_
  /-
    case intro.intro.intro.intro
    α : Type u_3
    ι : Type u_4
    m : MeasurableSpace α
    π : ι → Set (Set α)
    h : ∀ (n : ι), LE.le (MeasurableSpace.generateFrom (π n)) m
    S : Set ι
    ht_p : Finset ι
    w✝ : HasSubset.Subset (↑ht_p) S
    ft : ι → Set α
    hft_mem_pi : ∀ (x : ι), Membership.mem ht_p x → Membership.mem (π x) (ft x)
    x : ι
    hx_mem : Membership.mem ht_p x
    ⊢ MeasurableSet (ft x)
  -/
  exact measurableSet_generateFrom (hft_mem_pi x hx_mem)
  /-
    🎉 no goals
  -/


theorem subset_piiUnionInter {π : ι → Set (Set α)} {S : Set ι} {i : ι} (his : i ∈ S) :
    π i ⊆ piiUnionInter π S := by
  have h_ss : {i} ⊆ S := by
    intro j hj
    rw [mem_singleton_iff] at hj
    rwa [hj]
  /-
    α : Type u_3
    ι : Type u_4
    π : ι → Set (Set α)
    S : Set ι
    i : ι
    his : Membership.mem S i
    h_ss : HasSubset.Subset (Singleton.singleton i) S
    ⊢ HasSubset.Subset (π i) (piiUnionInter π S)
  -/
  refine Subset.trans ?_ (piiUnionInter_mono_right h_ss)
  /-
    α : Type u_3
    ι : Type u_4
    π : ι → Set (Set α)
    S : Set ι
    i : ι
    his : Membership.mem S i
    h_ss : HasSubset.Subset (Singleton.singleton i) S
    ⊢ HasSubset.Subset (π i) (piiUnionInter π (Singleton.singleton i))
  -/
  rw [piiUnionInter_singleton]
  /-
    α : Type u_3
    ι : Type u_4
    π : ι → Set (Set α)
    S : Set ι
    i : ι
    his : Membership.mem S i
    h_ss : HasSubset.Subset (Singleton.singleton i) S
    ⊢ HasSubset.Subset (π i) (Union.union (π i) (Singleton.singleton Set.univ))
  -/
  exact subset_union_left
  /-
    🎉 no goals
  -/


theorem mem_piiUnionInter_of_measurableSet (m : ι → MeasurableSpace α) {S : Set ι} {i : ι}
    (hiS : i ∈ S) (s : Set α) (hs : MeasurableSet[m i] s) :
    s ∈ piiUnionInter (fun n => { s | MeasurableSet[m n] s }) S :=
  subset_piiUnionInter hiS hs


theorem le_generateFrom_piiUnionInter {π : ι → Set (Set α)} (S : Set ι) {x : ι} (hxS : x ∈ S) :
    generateFrom (π x) ≤ generateFrom (piiUnionInter π S) :=
  generateFrom_mono (subset_piiUnionInter hxS)


theorem measurableSet_iSup_of_mem_piiUnionInter (m : ι → MeasurableSpace α) (S : Set ι) (t : Set α)
    (ht : t ∈ piiUnionInter (fun n => { s | MeasurableSet[m n] s }) S) :
    MeasurableSet[⨆ i ∈ S, m i] t := by
  /-
    α : Type u_3
    ι : Type u_4
    m : ι → MeasurableSpace α
    S : Set ι
    t : Set α
    ht : Membership.mem (piiUnionInter (fun n => setOf fun s => MeasurableSet s) S …
    ⊢ MeasurableSet t
  -/
  rcases ht with ⟨pt, hpt, ft, ht_m, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_3
    ι : Type u_4
    m : ι → MeasurableSpace α
    S : Set ι
    pt : Finset ι
    hpt : HasSubset.Subset (↑pt) S
    ft : ι → Set α
    ht_m : ∀ (x : ι), Membership.mem pt x → Membership.mem ((fun n => setOf fun s  …
    ⊢ MeasurableSet (Set.iInter fun x => Set.iInter fun h => ft x)
  -/
  refine pt.measurableSet_biInter fun i hi => ?_
  /-
    case intro.intro.intro.intro
    α : Type u_3
    ι : Type u_4
    m : ι → MeasurableSpace α
    S : Set ι
    pt : Finset ι
    hpt : HasSubset.Subset (↑pt) S
    ft : ι → Set α
    ht_m : ∀ (x : ι), Membership.mem pt x → Membership.mem ((fun n => setOf fun s  …
    i : ι
    hi : Membership.mem pt i
    ⊢ MeasurableSet (ft i)
  -/
  suffices h_le : m i ≤ ⨆ i ∈ S, m i from h_le (ft i) (ht_m i hi)
  /-
    case intro.intro.intro.intro
    α : Type u_3
    ι : Type u_4
    m : ι → MeasurableSpace α
    S : Set ι
    pt : Finset ι
    hpt : HasSubset.Subset (↑pt) S
    ft : ι → Set α
    ht_m : ∀ (x : ι), Membership.mem pt x → Membership.mem ((fun n => setOf fun s  …
    i : ι
    hi : Membership.mem pt i
    ⊢ LE.le (m i) (iSup fun i => iSup fun h => m i)
  -/
  have hi' : i ∈ S := hpt hi
  /-
    case intro.intro.intro.intro
    α : Type u_3
    ι : Type u_4
    m : ι → MeasurableSpace α
    S : Set ι
    pt : Finset ι
    hpt : HasSubset.Subset (↑pt) S
    ft : ι → Set α
    ht_m : ∀ (x : ι), Membership.mem pt x → Membership.mem ((fun n => setOf fun s  …
    i : ι
    hi : Membership.mem pt i
    hi' : Membership.mem S i
    ⊢ LE.le (m i) (iSup fun i => iSup fun h => m i)
  -/
  exact le_iSup₂ (f := fun i (_ : i ∈ S) => m i) i hi'
  /-
    🎉 no goals
  -/


theorem generateFrom_piiUnionInter_measurableSet (m : ι → MeasurableSpace α) (S : Set ι) :
    generateFrom (piiUnionInter (fun n => { s | MeasurableSet[m n] s }) S) = ⨆ i ∈ S, m i := by
  /-
    α : Type u_3
    ι : Type u_4
    m : ι → MeasurableSpace α
    S : Set ι
    ⊢ Eq (MeasurableSpace.generateFrom (piiUnionInter (fun n => setOf fun s => Mea …
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      α : Type u_3
      ι : Type u_4
      m : ι → MeasurableSpace α
      S : Set ι
      ⊢ LE.le (MeasurableSpace.generateFrom (piiUnionInter (fun n => setOf fun s =>  …
    -/
  · rw [← @generateFrom_measurableSet α (⨆ i ∈ S, m i)]
    /-
      case refine_1
      α : Type u_3
      ι : Type u_4
      m : ι → MeasurableSpace α
      S : Set ι
      ⊢ LE.le (MeasurableSpace.generateFrom (piiUnionInter (fun n => setOf fun s =>  …
    -/
    exact generateFrom_mono (measurableSet_iSup_of_mem_piiUnionInter m S)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_3
      ι : Type u_4
      m : ι → MeasurableSpace α
      S : Set ι
      ⊢ LE.le (iSup fun i => iSup fun h => m i) (MeasurableSpace.generateFrom (piiUn …
    -/
  · refine iSup₂_le fun i hi => ?_
    /-
      case refine_2
      α : Type u_3
      ι : Type u_4
      m : ι → MeasurableSpace α
      S : Set ι
      i : ι
      hi : Membership.mem S i
      ⊢ LE.le (m i) (MeasurableSpace.generateFrom (piiUnionInter (fun n => setOf fun …
    -/
    rw [← @generateFrom_measurableSet α (m i)]
    /-
      case refine_2
      α : Type u_3
      ι : Type u_4
      m : ι → MeasurableSpace α
      S : Set ι
      i : ι
      hi : Membership.mem S i
      ⊢ LE.le (MeasurableSpace.generateFrom (setOf fun s => MeasurableSet s)) (Measu …
    -/
    exact generateFrom_mono (mem_piiUnionInter_of_measurableSet m hi)
    /-
      🎉 no goals
    -/


/-- A Dynkin system is a collection of subsets of a type `α` that contains the empty set,
  is closed under complementation and under countable union of pairwise disjoint sets.
  The disjointness condition is the only difference with `σ`-algebras.

  The main purpose of Dynkin systems is to provide a powerful induction rule for σ-algebras
  generated by a collection of sets which is stable under intersection.

  A Dynkin system is also known as a "λ-system" or a "d-system".
-/
structure DynkinSystem (α : Type*) where
  /-- Predicate saying that a given set is contained in the Dynkin system. -/
  Has : Set α → Prop
  /-- A Dynkin system contains the empty set. -/
  has_empty : Has ∅
  /-- A Dynkin system is closed under complementation. -/
  has_compl : ∀ {a}, Has a → Has aᶜ
  /-- A Dynkin system is closed under countable union of pairwise disjoint sets. Use a more general
  `MeasurableSpace.DynkinSystem.has_iUnion` instead. -/
  has_iUnion_nat : ∀ {f : ℕ → Set α}, Pairwise (Disjoint on f) → (∀ i, Has (f i)) → Has (⋃ i, f i)


@[ext]
theorem ext : ∀ {d₁ d₂ : DynkinSystem α}, (∀ s : Set α, d₁.Has s ↔ d₂.Has s) → d₁ = d₂
  | ⟨s₁, _, _, _⟩, ⟨s₂, _, _, _⟩, h => by
    /-
      α : Type u_3
      s₁ : Set α → Prop
      has_empty✝¹ : s₁ EmptyCollection.emptyCollection
      has_compl✝¹ : ∀ {a : Set α}, s₁ a → s₁ (HasCompl.compl a)
      has_iUnion_nat✝¹ : ∀ {f : Nat → Set α}, Pairwise (Function.onFun Disjoint f) → …
      s₂ : Set α → Prop
      has_empty✝ : s₂ EmptyCollection.emptyCollection
      has_compl✝ : ∀ {a : Set α}, s₂ a → s₂ (HasCompl.compl a)
      has_iUnion_nat✝ : ∀ {f : Nat → Set α}, Pairwise (Function.onFun Disjoint f) →  …
      h : ∀ (s : Set α), Iff ({ Has := s₁, has_empty := has_empty✝¹, has_compl := ha …
      ⊢ Eq { Has := s₁, has_empty := has_empty✝¹, has_compl := has_compl✝¹, has_iUni …
    -/
    have : s₁ = s₂ := funext fun x => propext <| h x
    /-
      α : Type u_3
      s₁ : Set α → Prop
      has_empty✝¹ : s₁ EmptyCollection.emptyCollection
      has_compl✝¹ : ∀ {a : Set α}, s₁ a → s₁ (HasCompl.compl a)
      has_iUnion_nat✝¹ : ∀ {f : Nat → Set α}, Pairwise (Function.onFun Disjoint f) → …
      s₂ : Set α → Prop
      has_empty✝ : s₂ EmptyCollection.emptyCollection
      has_compl✝ : ∀ {a : Set α}, s₂ a → s₂ (HasCompl.compl a)
      has_iUnion_nat✝ : ∀ {f : Nat → Set α}, Pairwise (Function.onFun Disjoint f) →  …
      h : ∀ (s : Set α), Iff ({ Has := s₁, has_empty := has_empty✝¹, has_compl := ha …
      this : Eq s₁ s₂
      ⊢ Eq { Has := s₁, has_empty := has_empty✝¹, has_compl := has_compl✝¹, has_iUni …
    -/
    subst this
    /-
      α : Type u_3
      s₁ : Set α → Prop
      has_empty✝¹ : s₁ EmptyCollection.emptyCollection
      has_compl✝¹ : ∀ {a : Set α}, s₁ a → s₁ (HasCompl.compl a)
      has_iUnion_nat✝¹ : ∀ {f : Nat → Set α}, Pairwise (Function.onFun Disjoint f) → …
      has_empty✝ : s₁ EmptyCollection.emptyCollection
      has_compl✝ : ∀ {a : Set α}, s₁ a → s₁ (HasCompl.compl a)
      has_iUnion_nat✝ : ∀ {f : Nat → Set α}, Pairwise (Function.onFun Disjoint f) →  …
      h : ∀ (s : Set α), Iff ({ Has := s₁, has_empty := has_empty✝¹, has_compl := ha …
      ⊢ Eq { Has := s₁, has_empty := has_empty✝¹, has_compl := has_compl✝¹, has_iUni …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem has_compl_iff {a} : d.Has aᶜ ↔ d.Has a :=
               /-
                 α : Type u_3
                 d : MeasurableSpace.DynkinSystem α
                 a : Set α
                 h : d.Has (HasCompl.compl a)
                 ⊢ d.Has a
               -/
  ⟨fun h => by simpa using d.has_compl h, fun h => d.has_compl h⟩
               /-
                 🎉 no goals
               -/


                                    /-
                                      α : Type u_3
                                      d : MeasurableSpace.DynkinSystem α
                                      ⊢ d.Has Set.univ
                                    -/
theorem has_univ : d.Has univ := by simpa using d.has_compl d.has_empty
                                    /-
                                      🎉 no goals
                                    -/


theorem has_iUnion {β} [Countable β] {f : β → Set α} (hd : Pairwise (Disjoint on f))
    (h : ∀ i, d.Has (f i)) : d.Has (⋃ i, f i) := by
  /-
    α : Type u_3
    d : MeasurableSpace.DynkinSystem α
    β : Type u_4
    inst✝ : Countable β
    f : β → Set α
    hd : Pairwise (Function.onFun Disjoint f)
    h : ∀ (i : β), d.Has (f i)
    ⊢ d.Has (Set.iUnion fun i => f i)
  -/
  cases nonempty_encodable β
  /-
    case intro
    α : Type u_3
    d : MeasurableSpace.DynkinSystem α
    β : Type u_4
    inst✝ : Countable β
    f : β → Set α
    hd : Pairwise (Function.onFun Disjoint f)
    h : ∀ (i : β), d.Has (f i)
    val✝ : Encodable β
    ⊢ d.Has (Set.iUnion fun i => f i)
  -/
  rw [← Encodable.iUnion_decode₂]
  exact
    d.has_iUnion_nat (Encodable.iUnion_decode₂_disjoint_on hd) fun n =>
      Encodable.iUnion_decode₂_cases d.has_empty h


theorem has_union {s₁ s₂ : Set α} (h₁ : d.Has s₁) (h₂ : d.Has s₂) (h : Disjoint s₁ s₂) :
    d.Has (s₁ ∪ s₂) := by
  /-
    α : Type u_3
    d : MeasurableSpace.DynkinSystem α
    s₁ s₂ : Set α
    h₁ : d.Has s₁
    h₂ : d.Has s₂
    h : Disjoint s₁ s₂
    ⊢ d.Has (Union.union s₁ s₂)
  -/
  rw [union_eq_iUnion]
  /-
    α : Type u_3
    d : MeasurableSpace.DynkinSystem α
    s₁ s₂ : Set α
    h₁ : d.Has s₁
    h₂ : d.Has s₂
    h : Disjoint s₁ s₂
    ⊢ d.Has (Set.iUnion fun b => cond b s₁ s₂)
  -/
  exact d.has_iUnion (pairwise_disjoint_on_bool.2 h) (Bool.forall_bool.2 ⟨h₂, h₁⟩)
  /-
    🎉 no goals
  -/


theorem has_diff {s₁ s₂ : Set α} (h₁ : d.Has s₁) (h₂ : d.Has s₂) (h : s₂ ⊆ s₁) :
    d.Has (s₁ \ s₂) := by
  /-
    α : Type u_3
    d : MeasurableSpace.DynkinSystem α
    s₁ s₂ : Set α
    h₁ : d.Has s₁
    h₂ : d.Has s₂
    h : HasSubset.Subset s₂ s₁
    ⊢ d.Has (SDiff.sdiff s₁ s₂)
  -/
  apply d.has_compl_iff.1
  /-
    α : Type u_3
    d : MeasurableSpace.DynkinSystem α
    s₁ s₂ : Set α
    h₁ : d.Has s₁
    h₂ : d.Has s₂
    h : HasSubset.Subset s₂ s₁
    ⊢ d.Has (HasCompl.compl (SDiff.sdiff s₁ s₂))
  -/
  simp only [diff_eq, compl_inter, compl_compl]
  /-
    α : Type u_3
    d : MeasurableSpace.DynkinSystem α
    s₁ s₂ : Set α
    h₁ : d.Has s₁
    h₂ : d.Has s₂
    h : HasSubset.Subset s₂ s₁
    ⊢ d.Has (Union.union (HasCompl.compl s₁) s₂)
  -/
  exact d.has_union (d.has_compl h₁) h₂ (disjoint_compl_left.mono_right h)
  /-
    🎉 no goals
  -/


instance instLEDynkinSystem : LE (DynkinSystem α) where le m₁ m₂ := m₁.Has ≤ m₂.Has


theorem le_def {a b : DynkinSystem α} : a ≤ b ↔ a.Has ≤ b.Has :=
  Iff.rfl


instance : PartialOrder (DynkinSystem α) :=
  { DynkinSystem.instLEDynkinSystem with
    le_refl := fun _ _ => le_rfl
    le_trans := fun _ _ _ hab hbc => le_def.mpr (le_trans hab hbc)
    le_antisymm := fun _ _ h₁ h₂ => ext fun s => ⟨h₁ s, h₂ s⟩ }


/-- Every measurable space (σ-algebra) forms a Dynkin system -/
def ofMeasurableSpace (m : MeasurableSpace α) : DynkinSystem α where
  Has := m.MeasurableSet'
  has_empty := m.measurableSet_empty
  has_compl {a} := m.measurableSet_compl a
  has_iUnion_nat {f} _ hf := m.measurableSet_iUnion f hf


theorem ofMeasurableSpace_le_ofMeasurableSpace_iff {m₁ m₂ : MeasurableSpace α} :
    ofMeasurableSpace m₁ ≤ ofMeasurableSpace m₂ ↔ m₁ ≤ m₂ :=
  Iff.rfl


/-- The least Dynkin system containing a collection of basic sets.
  This inductive type gives the underlying collection of sets. -/
inductive GenerateHas (s : Set (Set α)) : Set α → Prop
  | basic : ∀ t ∈ s, GenerateHas s t
  | empty : GenerateHas s ∅
  | compl : ∀ {a}, GenerateHas s a → GenerateHas s aᶜ
  | iUnion : ∀ {f : ℕ → Set α},
    Pairwise (Disjoint on f) → (∀ i, GenerateHas s (f i)) → GenerateHas s (⋃ i, f i)


theorem generateHas_compl {C : Set (Set α)} {s : Set α} : GenerateHas C sᶜ ↔ GenerateHas C s := by
  /-
    α : Type u_3
    C : Set (Set α)
    s : Set α
    ⊢ Iff (MeasurableSpace.DynkinSystem.GenerateHas C (HasCompl.compl s)) (Measura …
  -/
  refine ⟨?_, GenerateHas.compl⟩
  /-
    α : Type u_3
    C : Set (Set α)
    s : Set α
    ⊢ MeasurableSpace.DynkinSystem.GenerateHas C (HasCompl.compl s) → MeasurableSp …
  -/
  intro h
  /-
    α : Type u_3
    C : Set (Set α)
    s : Set α
    h : MeasurableSpace.DynkinSystem.GenerateHas C (HasCompl.compl s)
    ⊢ MeasurableSpace.DynkinSystem.GenerateHas C s
  -/
  convert GenerateHas.compl h
  /-
    case h.e'_3
    α : Type u_3
    C : Set (Set α)
    s : Set α
    h : MeasurableSpace.DynkinSystem.GenerateHas C (HasCompl.compl s)
    ⊢ Eq s (HasCompl.compl (HasCompl.compl s))
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The least Dynkin system containing a collection of basic sets. -/
def generate (s : Set (Set α)) : DynkinSystem α where
  Has := GenerateHas s
  has_empty := GenerateHas.empty
  has_compl {_} := GenerateHas.compl
  has_iUnion_nat {_} := GenerateHas.iUnion


theorem generateHas_def {C : Set (Set α)} : (generate C).Has = GenerateHas C :=
  rfl


instance : Inhabited (DynkinSystem α) :=
  ⟨generate univ⟩


/-- If a Dynkin system is closed under binary intersection, then it forms a `σ`-algebra. -/
def toMeasurableSpace (h_inter : ∀ s₁ s₂, d.Has s₁ → d.Has s₂ → d.Has (s₁ ∩ s₂)) :
    MeasurableSpace α where
  MeasurableSet' := d.Has
  measurableSet_empty := d.has_empty
  measurableSet_compl _ h := d.has_compl h
  measurableSet_iUnion f hf := by
    /-
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      d : MeasurableSpace.DynkinSystem α
      h_inter : ∀ (s₁ s₂ : Set α), d.Has s₁ → d.Has s₂ → d.Has (Inter.inter s₁ s₂)
      f : Nat → Set α
      hf : ∀ (i : Nat), d.Has (f i)
      ⊢ d.Has (Set.iUnion fun i => f i)
    -/
    rw [← iUnion_disjointed]
    exact
      d.has_iUnion (disjoint_disjointed _) fun n =>
        disjointedRec (fun (t : Set α) i h => h_inter _ _ h <| d.has_compl <| hf i) (hf n)


theorem ofMeasurableSpace_toMeasurableSpace
    (h_inter : ∀ s₁ s₂, d.Has s₁ → d.Has s₂ → d.Has (s₁ ∩ s₂)) :
    ofMeasurableSpace (d.toMeasurableSpace h_inter) = d :=
  ext fun _ => Iff.rfl


/-- If `s` is in a Dynkin system `d`, we can form the new Dynkin system `{s ∩ t | t ∈ d}`. -/
def restrictOn {s : Set α} (h : d.Has s) : DynkinSystem α where
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
  Has t := d.Has (t ∩ s)
                  /-
                    α✝ : Type u_1
                    β : Type u_2
                    α : Type u_3
                    d : MeasurableSpace.DynkinSystem α
                    s : Set α
                    h : d.Has s
                    ⊢ (fun t => d.Has (Inter.inter t s)) EmptyCollection.emptyCollection
                  -/
  has_empty := by simp [d.has_empty]
                  /-
                    🎉 no goals
                  -/
  has_compl {t} hts := by
    /-
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      d : MeasurableSpace.DynkinSystem α
      s : Set α
      h : d.Has s
      t : Set α
      hts : (fun t => d.Has (Inter.inter t s)) t
      ⊢ (fun t => d.Has (Inter.inter t s)) (HasCompl.compl t)
    -/
    beta_reduce
    /-
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      d : MeasurableSpace.DynkinSystem α
      s : Set α
      h : d.Has s
      t : Set α
      hts : (fun t => d.Has (Inter.inter t s)) t
      ⊢ d.Has (Inter.inter (HasCompl.compl t) s)
    -/
    have : tᶜ ∩ s = (t ∩ s)ᶜ \ sᶜ := Set.ext fun x => by by_cases h : x ∈ s <;> simp [h]
    /-
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      d : MeasurableSpace.DynkinSystem α
      s : Set α
      h : d.Has s
      t : Set α
      hts : (fun t => d.Has (Inter.inter t s)) t
      this : Eq (Inter.inter (HasCompl.compl t) s) (SDiff.sdiff (HasCompl.compl (Int …
      ⊢ d.Has (Inter.inter (HasCompl.compl t) s)
    -/
    rw [this]
    exact
      d.has_diff (d.has_compl hts) (d.has_compl h)
        (compl_subset_compl.mpr inter_subset_right)
  has_iUnion_nat {f} hd hf := by
    /-
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      d : MeasurableSpace.DynkinSystem α
      s : Set α
      h : d.Has s
      f : Nat → Set α
      hd : Pairwise (Function.onFun Disjoint f)
      hf : ∀ (i : Nat), (fun t => d.Has (Inter.inter t s)) (f i)
      ⊢ (fun t => d.Has (Inter.inter t s)) (Set.iUnion fun i => f i)
    -/
    simp only []
    /-
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      d : MeasurableSpace.DynkinSystem α
      s : Set α
      h : d.Has s
      f : Nat → Set α
      hd : Pairwise (Function.onFun Disjoint f)
      hf : ∀ (i : Nat), (fun t => d.Has (Inter.inter t s)) (f i)
      ⊢ d.Has (Inter.inter (Set.iUnion fun i => f i) s)
    -/
    rw [iUnion_inter]
    /-
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      d : MeasurableSpace.DynkinSystem α
      s : Set α
      h : d.Has s
      f : Nat → Set α
      hd : Pairwise (Function.onFun Disjoint f)
      hf : ∀ (i : Nat), (fun t => d.Has (Inter.inter t s)) (f i)
      ⊢ d.Has (Set.iUnion fun i => Inter.inter (f i) s)
    -/
    refine d.has_iUnion_nat ?_ hf
    /-
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      d : MeasurableSpace.DynkinSystem α
      s : Set α
      h : d.Has s
      f : Nat → Set α
      hd : Pairwise (Function.onFun Disjoint f)
      hf : ∀ (i : Nat), (fun t => d.Has (Inter.inter t s)) (f i)
      ⊢ Pairwise (Function.onFun Disjoint fun i => Inter.inter (f i) s)
    -/
    exact hd.mono fun i j => Disjoint.mono inter_subset_left inter_subset_left
    /-
      🎉 no goals
    -/


theorem generate_le {s : Set (Set α)} (h : ∀ t ∈ s, d.Has t) : generate s ≤ d := fun _ ht =>
  ht.recOn h d.has_empty (fun {_} _ h => d.has_compl h) fun {_} hd _ hf => d.has_iUnion hd hf


theorem generate_has_subset_generate_measurable {C : Set (Set α)} {s : Set α}
    (hs : (generate C).Has s) : MeasurableSet[generateFrom C] s :=
  generate_le (ofMeasurableSpace (generateFrom C)) (fun _ => measurableSet_generateFrom) s hs


theorem generate_inter {s : Set (Set α)} (hs : IsPiSystem s) {t₁ t₂ : Set α}
    (ht₁ : (generate s).Has t₁) (ht₂ : (generate s).Has t₂) : (generate s).Has (t₁ ∩ t₂) :=
  have : generate s ≤ (generate s).restrictOn ht₂ :=
    generate_le _ fun s₁ hs₁ =>
      have : (generate s).Has s₁ := GenerateHas.basic s₁ hs₁
      have : generate s ≤ (generate s).restrictOn this :=
        generate_le _ fun s₂ hs₂ =>
          show (generate s).Has (s₂ ∩ s₁) from
            (s₂ ∩ s₁).eq_empty_or_nonempty.elim (fun h => h.symm ▸ GenerateHas.empty) fun h =>
              GenerateHas.basic _ <| hs _ hs₂ _ hs₁ h
      have : (generate s).Has (t₂ ∩ s₁) := this _ ht₂
                                         /-
                                           α : Type u_3
                                           s : Set (Set α)
                                           hs : IsPiSystem s
                                           t₁ t₂ : Set α
                                           ht₁ : (MeasurableSpace.DynkinSystem.generate s).Has t₁
                                           ht₂ : (MeasurableSpace.DynkinSystem.generate s).Has t₂
                                           s₁ : Set α
                                           hs₁ : Membership.mem s s₁
                                           this✝¹ : (MeasurableSpace.DynkinSystem.generate s).Has s₁
                                           this✝ : LE.le (MeasurableSpace.DynkinSystem.generate s) ((MeasurableSpace.Dynk …
                                           this : (MeasurableSpace.DynkinSystem.generate s).Has (Inter.inter t₂ s₁)
                                           ⊢ (MeasurableSpace.DynkinSystem.generate s).Has (Inter.inter s₁ t₂)
                                         -/
      show (generate s).Has (s₁ ∩ t₂) by rwa [inter_comm]
                                         /-
                                           🎉 no goals
                                         -/
  this _ ht₁


/-- **Dynkin's π-λ theorem**:
  Given a collection of sets closed under binary intersections, then the Dynkin system it
  generates is equal to the σ-algebra it generates.
  This result is known as the π-λ theorem.
  A collection of sets closed under binary intersection is called a π-system (often requiring
  additionally that it is non-empty, but we drop this condition in the formalization).
-/
theorem generateFrom_eq {s : Set (Set α)} (hs : IsPiSystem s) :
    generateFrom s = (generate s).toMeasurableSpace fun _ _ => generate_inter hs :=
  le_antisymm (generateFrom_le fun t ht => GenerateHas.basic t ht)
    (ofMeasurableSpace_le_ofMeasurableSpace_iff.mp <| by
      /-
        α : Type u_3
        s : Set (Set α)
        hs : IsPiSystem s
        ⊢ LE.le (MeasurableSpace.DynkinSystem.ofMeasurableSpace ((MeasurableSpace.Dynk …
      -/
      rw [ofMeasurableSpace_toMeasurableSpace]
      /-
        α : Type u_3
        s : Set (Set α)
        hs : IsPiSystem s
        ⊢ LE.le (MeasurableSpace.DynkinSystem.generate s) (MeasurableSpace.DynkinSyste …
      -/
      exact generate_le _ fun t ht => measurableSet_generateFrom ht)
      /-
        🎉 no goals
      -/


/-- Induction principle for measurable sets.
If `s` is a π-system that generates the product `σ`-algebra on `α`
and a predicate `C` defined on measurable sets is true

- on the empty set;
- on each set `t ∈ s`;
- on the complement of a measurable set that satisfies `C`;
- on the union of a sequence of pairwise disjoint measurable sets that satisfy `C`,

then it is true on all measurable sets in `α`. -/
@[elab_as_elim]
theorem induction_on_inter {m : MeasurableSpace α} {C : ∀ s : Set α, MeasurableSet s → Prop}
    {s : Set (Set α)} (h_eq : m = generateFrom s) (h_inter : IsPiSystem s)
    (empty : C ∅ .empty) (basic : ∀ t (ht : t ∈ s), C t <| h_eq ▸ .basic t ht)
    (compl : ∀ t (htm : MeasurableSet t), C t htm → C tᶜ htm.compl)
    (iUnion : ∀ (f : ℕ → Set α), Pairwise (Disjoint on f) → ∀ (hfm : ∀ i, MeasurableSet (f i)),
      (∀ i, C (f i) (hfm i)) → C (⋃ i, f i) (.iUnion hfm)) :
    ∀ t (ht : MeasurableSet t), C t ht := by
  have eq : MeasurableSet = DynkinSystem.GenerateHas s := by
    rw [h_eq, DynkinSystem.generateFrom_eq h_inter]
    rfl
  suffices ∀ t (ht : DynkinSystem.GenerateHas s t), C t (eq ▸ ht) from
    fun t ht ↦ this t (eq ▸ ht)
  /-
    α : Type u_3
    m : MeasurableSpace α
    C : (s : Set α) → MeasurableSet s → Prop
    s : Set (Set α)
    h_eq : Eq m (MeasurableSpace.generateFrom s)
    h_inter : IsPiSystem s
    empty : C EmptyCollection.emptyCollection ⋯
    basic : ∀ (t : Set α) (ht : Membership.mem s t), C t ⋯
    compl : ∀ (t : Set α) (htm : MeasurableSet t), C t htm → C (HasCompl.compl t) ⋯
    iUnion : ∀ (f : Nat → Set α), Pairwise (Function.onFun Disjoint f) → ∀ (hfm :  …
    eq : Eq MeasurableSet (MeasurableSpace.DynkinSystem.GenerateHas s)
    ⊢ ∀ (t : Set α) (ht : MeasurableSpace.DynkinSystem.GenerateHas s t), C t ⋯
  -/
  intro t ht
  induction ht with
  | basic u hu => exact basic u hu
  | empty => exact empty
  | @compl u hu ihu => exact compl _ (eq ▸ hu) ihu
  | @iUnion f hfd hf ihf => exact iUnion f hfd (eq ▸ hf) ihf


