/-- The type of Heyting homomorphisms from `α` to `β`. Bounded lattice homomorphisms that preserve
Heyting implication. -/
structure HeytingHom (α β : Type*) [HeytingAlgebra α] [HeytingAlgebra β] extends
  LatticeHom α β where
  /-- The proposition that a Heyting homomorphism preserves the bottom element. -/
  protected map_bot' : toFun ⊥ = ⊥
  /-- The proposition that a Heyting homomorphism preserves the Heyting implication. -/
  protected map_himp' : ∀ a b, toFun (a ⇨ b) = toFun a ⇨ toFun b


/-- The type of co-Heyting homomorphisms from `α` to `β`. Bounded lattice homomorphisms that
preserve difference. -/
structure CoheytingHom (α β : Type*) [CoheytingAlgebra α] [CoheytingAlgebra β] extends
  LatticeHom α β where
  /-- The proposition that a co-Heyting homomorphism preserves the top element. -/
  protected map_top' : toFun ⊤ = ⊤
  /-- The proposition that a co-Heyting homomorphism preserves the difference operation. -/
  protected map_sdiff' : ∀ a b, toFun (a \ b) = toFun a \ toFun b


/-- The type of bi-Heyting homomorphisms from `α` to `β`. Bounded lattice homomorphisms that
preserve Heyting implication and difference. -/
structure BiheytingHom (α β : Type*) [BiheytingAlgebra α] [BiheytingAlgebra β] extends
  LatticeHom α β where
  /-- The proposition that a bi-Heyting homomorphism preserves the Heyting implication. -/
  protected map_himp' : ∀ a b, toFun (a ⇨ b) = toFun a ⇨ toFun b
  /-- The proposition that a bi-Heyting homomorphism preserves the difference operation. -/
  protected map_sdiff' : ∀ a b, toFun (a \ b) = toFun a \ toFun b


/-- `HeytingHomClass F α β` states that `F` is a type of Heyting homomorphisms.

You should extend this class when you extend `HeytingHom`. -/
class HeytingHomClass (F α β : Type*) [HeytingAlgebra α] [HeytingAlgebra β] [FunLike F α β]
  extends LatticeHomClass F α β : Prop where
  /-- The proposition that a Heyting homomorphism preserves the bottom element. -/
  map_bot (f : F) : f ⊥ = ⊥
  /-- The proposition that a Heyting homomorphism preserves the Heyting implication. -/
  map_himp (f : F) : ∀ a b, f (a ⇨ b) = f a ⇨ f b


/-- `CoheytingHomClass F α β` states that `F` is a type of co-Heyting homomorphisms.

You should extend this class when you extend `CoheytingHom`. -/
class CoheytingHomClass (F α β : Type*) [CoheytingAlgebra α] [CoheytingAlgebra β] [FunLike F α β]
  extends LatticeHomClass F α β : Prop where
  /-- The proposition that a co-Heyting homomorphism preserves the top element. -/
  map_top (f : F) : f ⊤ = ⊤
  /-- The proposition that a co-Heyting homomorphism preserves the difference operation. -/
  map_sdiff (f : F) : ∀ a b, f (a \ b) = f a \ f b


/-- `BiheytingHomClass F α β` states that `F` is a type of bi-Heyting homomorphisms.

You should extend this class when you extend `BiheytingHom`. -/
class BiheytingHomClass (F α β : Type*) [BiheytingAlgebra α] [BiheytingAlgebra β] [FunLike F α β]
  extends LatticeHomClass F α β : Prop where
  /-- The proposition that a bi-Heyting homomorphism preserves the Heyting implication. -/
  map_himp (f : F) : ∀ a b, f (a ⇨ b) = f a ⇨ f b
  /-- The proposition that a bi-Heyting homomorphism preserves the difference operation. -/
  map_sdiff (f : F) : ∀ a b, f (a \ b) = f a \ f b


instance (priority := 100) HeytingHomClass.toBoundedLatticeHomClass [HeytingAlgebra α]
    { _ : HeytingAlgebra β} [HeytingHomClass F α β] : BoundedLatticeHomClass F α β :=
  { ‹HeytingHomClass F α β› with
                           /-
                             F : Type u_1
                             α : Type u_2
                             β : Type u_3
                             γ : Type u_4
                             δ : Type u_5
                             inst✝² : FunLike F α β
                             inst✝¹ : HeytingAlgebra α
                             x✝ : HeytingAlgebra β
                             inst✝ : HeytingHomClass F α β
                             f : F
                             ⊢ Eq (f Top.top) Top.top
                           -/
    map_top := fun f => by rw [← @himp_self α _ ⊥, ← himp_self, map_himp] }
                           /-
                             🎉 no goals
                           -/

-- See note [lower instance priority]

instance (priority := 100) CoheytingHomClass.toBoundedLatticeHomClass [CoheytingAlgebra α]
    { _ : CoheytingAlgebra β} [CoheytingHomClass F α β] : BoundedLatticeHomClass F α β :=
  { ‹CoheytingHomClass F α β› with
                           /-
                             F : Type u_1
                             α : Type u_2
                             β : Type u_3
                             γ : Type u_4
                             δ : Type u_5
                             inst✝² : FunLike F α β
                             inst✝¹ : CoheytingAlgebra α
                             x✝ : CoheytingAlgebra β
                             inst✝ : CoheytingHomClass F α β
                             f : F
                             ⊢ Eq (f Bot.bot) Bot.bot
                           -/
    map_bot := fun f => by rw [← @sdiff_self α _ ⊤, ← sdiff_self, map_sdiff] }
                           /-
                             🎉 no goals
                           -/

-- See note [lower instance priority]

instance (priority := 100) BiheytingHomClass.toHeytingHomClass [BiheytingAlgebra α]
    { _ : BiheytingAlgebra β} [BiheytingHomClass F α β] : HeytingHomClass F α β :=
  { ‹BiheytingHomClass F α β› with
                           /-
                             F : Type u_1
                             α : Type u_2
                             β : Type u_3
                             γ : Type u_4
                             δ : Type u_5
                             inst✝² : FunLike F α β
                             inst✝¹ : BiheytingAlgebra α
                             x✝ : BiheytingAlgebra β
                             inst✝ : BiheytingHomClass F α β
                             f : F
                             ⊢ Eq (f Bot.bot) Bot.bot
                           -/
    map_bot := fun f => by rw [← @sdiff_self α _ ⊤, ← sdiff_self, BiheytingHomClass.map_sdiff] }
                           /-
                             🎉 no goals
                           -/

-- See note [lower instance priority]

instance (priority := 100) BiheytingHomClass.toCoheytingHomClass [BiheytingAlgebra α]
    { _ : BiheytingAlgebra β} [BiheytingHomClass F α β] : CoheytingHomClass F α β :=
  { ‹BiheytingHomClass F α β› with
                           /-
                             F : Type u_1
                             α : Type u_2
                             β : Type u_3
                             γ : Type u_4
                             δ : Type u_5
                             inst✝² : FunLike F α β
                             inst✝¹ : BiheytingAlgebra α
                             x✝ : BiheytingAlgebra β
                             inst✝ : BiheytingHomClass F α β
                             f : F
                             ⊢ Eq (f Top.top) Top.top
                           -/
    map_top := fun f => by rw [← @himp_self α _ ⊥, ← himp_self, map_himp] }
                           /-
                             🎉 no goals
                           -/


instance (priority := 100) OrderIsoClass.toHeytingHomClass [HeytingAlgebra α]
    { _ : HeytingAlgebra β} [OrderIsoClass F α β] : HeytingHomClass F α β :=
  { OrderIsoClass.toBoundedLatticeHomClass with
    map_himp := fun f a b =>
      eq_of_forall_le_iff fun c => by
        /-
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          δ : Type u_5
          inst✝² : EquivLike F α β
          inst✝¹ : HeytingAlgebra α
          x✝ : HeytingAlgebra β
          inst✝ : OrderIsoClass F α β
          f : F
          a b : α
          c : β
          ⊢ Iff (LE.le c (f (HImp.himp a b))) (LE.le c (HImp.himp (f a) (f b)))
        -/
        simp only [← map_inv_le_iff, le_himp_iff]
        /-
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          δ : Type u_5
          inst✝² : EquivLike F α β
          inst✝¹ : HeytingAlgebra α
          x✝ : HeytingAlgebra β
          inst✝ : OrderIsoClass F α β
          f : F
          a b : α
          c : β
          ⊢ Iff (LE.le (Min.min (EquivLike.inv f c) a) b) (LE.le (EquivLike.inv f (Min.m …
        -/
        rw [← OrderIsoClass.map_le_map_iff f]
        /-
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          δ : Type u_5
          inst✝² : EquivLike F α β
          inst✝¹ : HeytingAlgebra α
          x✝ : HeytingAlgebra β
          inst✝ : OrderIsoClass F α β
          f : F
          a b : α
          c : β
          ⊢ Iff (LE.le (f (Min.min (EquivLike.inv f c) a)) (f b)) (LE.le (EquivLike.inv  …
        -/
        simp }
        /-
          🎉 no goals
        -/

-- See note [lower instance priority]

instance (priority := 100) OrderIsoClass.toCoheytingHomClass [CoheytingAlgebra α]
    { _ : CoheytingAlgebra β} [OrderIsoClass F α β] : CoheytingHomClass F α β :=
  { OrderIsoClass.toBoundedLatticeHomClass with
    map_sdiff := fun f a b =>
      eq_of_forall_ge_iff fun c => by
        /-
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          δ : Type u_5
          inst✝² : EquivLike F α β
          inst✝¹ : CoheytingAlgebra α
          x✝ : CoheytingAlgebra β
          inst✝ : OrderIsoClass F α β
          f : F
          a b : α
          c : β
          ⊢ Iff (LE.le (f (SDiff.sdiff a b)) c) (LE.le (SDiff.sdiff (f a) (f b)) c)
        -/
        simp only [← le_map_inv_iff, sdiff_le_iff]
        /-
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          δ : Type u_5
          inst✝² : EquivLike F α β
          inst✝¹ : CoheytingAlgebra α
          x✝ : CoheytingAlgebra β
          inst✝ : OrderIsoClass F α β
          f : F
          a b : α
          c : β
          ⊢ Iff (LE.le a (Max.max b (EquivLike.inv f c))) (LE.le a (EquivLike.inv f (Max …
        -/
        rw [← OrderIsoClass.map_le_map_iff f]
        /-
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          δ : Type u_5
          inst✝² : EquivLike F α β
          inst✝¹ : CoheytingAlgebra α
          x✝ : CoheytingAlgebra β
          inst✝ : OrderIsoClass F α β
          f : F
          a b : α
          c : β
          ⊢ Iff (LE.le (f a) (f (Max.max b (EquivLike.inv f c)))) (LE.le a (EquivLike.in …
        -/
        simp }
        /-
          🎉 no goals
        -/

-- See note [lower instance priority]

instance (priority := 100) OrderIsoClass.toBiheytingHomClass [BiheytingAlgebra α]
    { _ : BiheytingAlgebra β} [OrderIsoClass F α β] : BiheytingHomClass F α β :=
  { OrderIsoClass.toLatticeHomClass with
    map_himp := fun f a b =>
      eq_of_forall_le_iff fun c => by
        /-
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          δ : Type u_5
          inst✝² : EquivLike F α β
          inst✝¹ : BiheytingAlgebra α
          x✝ : BiheytingAlgebra β
          inst✝ : OrderIsoClass F α β
          f : F
          a b : α
          c : β
          ⊢ Iff (LE.le c (f (HImp.himp a b))) (LE.le c (HImp.himp (f a) (f b)))
        -/
        simp only [← map_inv_le_iff, le_himp_iff]
        /-
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          δ : Type u_5
          inst✝² : EquivLike F α β
          inst✝¹ : BiheytingAlgebra α
          x✝ : BiheytingAlgebra β
          inst✝ : OrderIsoClass F α β
          f : F
          a b : α
          c : β
          ⊢ Iff (LE.le (Min.min (EquivLike.inv f c) a) b) (LE.le (EquivLike.inv f (Min.m …
        -/
        rw [← OrderIsoClass.map_le_map_iff f]
        /-
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          δ : Type u_5
          inst✝² : EquivLike F α β
          inst✝¹ : BiheytingAlgebra α
          x✝ : BiheytingAlgebra β
          inst✝ : OrderIsoClass F α β
          f : F
          a b : α
          c : β
          ⊢ Iff (LE.le (f (Min.min (EquivLike.inv f c) a)) (f b)) (LE.le (EquivLike.inv  …
        -/
        simp
        /-
          🎉 no goals
        -/
    map_sdiff := fun f a b =>
      eq_of_forall_ge_iff fun c => by
        /-
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          δ : Type u_5
          inst✝² : EquivLike F α β
          inst✝¹ : BiheytingAlgebra α
          x✝ : BiheytingAlgebra β
          inst✝ : OrderIsoClass F α β
          f : F
          a b : α
          c : β
          ⊢ Iff (LE.le (f (SDiff.sdiff a b)) c) (LE.le (SDiff.sdiff (f a) (f b)) c)
        -/
        simp only [← le_map_inv_iff, sdiff_le_iff]
        /-
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          δ : Type u_5
          inst✝² : EquivLike F α β
          inst✝¹ : BiheytingAlgebra α
          x✝ : BiheytingAlgebra β
          inst✝ : OrderIsoClass F α β
          f : F
          a b : α
          c : β
          ⊢ Iff (LE.le a (Max.max b (EquivLike.inv f c))) (LE.le a (EquivLike.inv f (Max …
        -/
        rw [← OrderIsoClass.map_le_map_iff f]
        /-
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          δ : Type u_5
          inst✝² : EquivLike F α β
          inst✝¹ : BiheytingAlgebra α
          x✝ : BiheytingAlgebra β
          inst✝ : OrderIsoClass F α β
          f : F
          a b : α
          c : β
          ⊢ Iff (LE.le (f a) (f (Max.max b (EquivLike.inv f c)))) (LE.le a (EquivLike.in …
        -/
        simp }
        /-
          🎉 no goals
        -/


/-- This can't be an instance because of typeclass loops. -/
lemma BoundedLatticeHomClass.toBiheytingHomClass [BooleanAlgebra α] [BooleanAlgebra β]
    [BoundedLatticeHomClass F α β] : BiheytingHomClass F α β :=
  { ‹BoundedLatticeHomClass F α β› with
                                /-
                                  F : Type u_1
                                  α : Type u_2
                                  β : Type u_3
                                  inst✝³ : FunLike F α β
                                  inst✝² : BooleanAlgebra α
                                  inst✝¹ : BooleanAlgebra β
                                  inst✝ : BoundedLatticeHomClass F α β
                                  f : F
                                  a b : α
                                  ⊢ Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
                                -/
    map_himp := fun f a b => by rw [himp_eq, himp_eq, map_sup, (isCompl_compl.map _).compl_eq]
                                /-
                                  🎉 no goals
                                -/
                                 /-
                                   F : Type u_1
                                   α : Type u_2
                                   β : Type u_3
                                   inst✝³ : FunLike F α β
                                   inst✝² : BooleanAlgebra α
                                   inst✝¹ : BooleanAlgebra β
                                   inst✝ : BoundedLatticeHomClass F α β
                                   f : F
                                   a b : α
                                   ⊢ Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
                                 -/
    map_sdiff := fun f a b => by rw [sdiff_eq, sdiff_eq, map_inf, (isCompl_compl.map _).compl_eq] }
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
                                                /-
                                                  F : Type u_1
                                                  α : Type u_2
                                                  β : Type u_3
                                                  inst✝³ : FunLike F α β
                                                  inst✝² : HeytingAlgebra α
                                                  inst✝¹ : HeytingAlgebra β
                                                  inst✝ : HeytingHomClass F α β
                                                  f : F
                                                  a : α
                                                  ⊢ Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
                                                -/
theorem map_compl (a : α) : f aᶜ = (f a)ᶜ := by rw [← himp_bot, ← himp_bot, map_himp, map_bot]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
                                                           /-
                                                             F : Type u_1
                                                             α : Type u_2
                                                             β : Type u_3
                                                             inst✝³ : FunLike F α β
                                                             inst✝² : HeytingAlgebra α
                                                             inst✝¹ : HeytingAlgebra β
                                                             inst✝ : HeytingHomClass F α β
                                                             f : F
                                                             a b : α
                                                             ⊢ Eq (f (bihimp a b)) (bihimp (f a) (f b))
                                                           -/
theorem map_bihimp (a b : α) : f (a ⇔ b) = f a ⇔ f b := by simp_rw [bihimp, map_inf, map_himp]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
                                               /-
                                                 F : Type u_1
                                                 α : Type u_2
                                                 β : Type u_3
                                                 inst✝³ : FunLike F α β
                                                 inst✝² : CoheytingAlgebra α
                                                 inst✝¹ : CoheytingAlgebra β
                                                 inst✝ : CoheytingHomClass F α β
                                                 f : F
                                                 a : α
                                                 ⊢ Eq (f (HNot.hnot a)) (HNot.hnot (f a))
                                               -/
theorem map_hnot (a : α) : f (￢a) = ￢f a := by rw [← top_sdiff', ← top_sdiff', map_sdiff, map_top]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
                                                             /-
                                                               F : Type u_1
                                                               α : Type u_2
                                                               β : Type u_3
                                                               inst✝³ : FunLike F α β
                                                               inst✝² : CoheytingAlgebra α
                                                               inst✝¹ : CoheytingAlgebra β
                                                               inst✝ : CoheytingHomClass F α β
                                                               f : F
                                                               a b : α
                                                               ⊢ Eq (f (symmDiff a b)) (symmDiff (f a) (f b))
                                                             -/
theorem map_symmDiff (a b : α) : f (a ∆ b) = f a ∆ f b := by simp_rw [symmDiff, map_sup, map_sdiff]
                                                             /-
                                                               🎉 no goals
                                                             -/


instance [HeytingAlgebra α] [HeytingAlgebra β] [HeytingHomClass F α β] : CoeTC F (HeytingHom α β) :=
  ⟨fun f =>
    { toFun := f
      map_sup' := map_sup f
      map_inf' := map_inf f
      map_bot' := map_bot f
      map_himp' := map_himp f }⟩


instance [CoheytingAlgebra α] [CoheytingAlgebra β] [CoheytingHomClass F α β] :
    CoeTC F (CoheytingHom α β) :=
  ⟨fun f =>
    { toFun := f
      map_sup' := map_sup f
      map_inf' := map_inf f
      map_top' := map_top f
      map_sdiff' := map_sdiff f }⟩


instance [BiheytingAlgebra α] [BiheytingAlgebra β] [BiheytingHomClass F α β] :
    CoeTC F (BiheytingHom α β) :=
  ⟨fun f =>
    { toFun := f
      map_sup' := map_sup f
      map_inf' := map_inf f
      map_himp' := map_himp f
      map_sdiff' := map_sdiff f }⟩


instance instFunLike : FunLike (HeytingHom α β) α β where
  coe f := f.toFun
                             /-
                               F : Type u_1
                               α : Type u_2
                               β : Type u_3
                               γ : Type u_4
                               δ : Type u_5
                               inst✝⁴ : FunLike F α β
                               inst✝³ : HeytingAlgebra α
                               inst✝² : HeytingAlgebra β
                               inst✝¹ : HeytingAlgebra γ
                               inst✝ : HeytingAlgebra δ
                               f g : HeytingHom α β
                               h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by obtain ⟨⟨⟨_, _⟩, _⟩, _⟩ := f; obtain ⟨⟨⟨_, _⟩, _⟩, _⟩ := g; congr
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


instance instHeytingHomClass : HeytingHomClass (HeytingHom α β) α β where
  map_sup f := f.map_sup'
  map_inf f := f.map_inf'
  map_bot f := f.map_bot'
  map_himp := HeytingHom.map_himp'

-- @[simp] -- Porting note: not in simp-nf, simp can simplify lhs. Added aux simp lemma

theorem toFun_eq_coe {f : HeytingHom α β} : f.toFun = ⇑f :=
  rfl


@[simp]
theorem toFun_eq_coe_aux {f : HeytingHom α β} : (↑f.toLatticeHom) = ⇑f :=
  rfl


@[ext]
theorem ext {f g : HeytingHom α β} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


/-- Copy of a `HeytingHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (f : HeytingHom α β) (f' : α → β) (h : f' = f) : HeytingHom α β where
  toFun := f'
                 /-
                   F : Type u_1
                   α : Type u_2
                   β : Type u_3
                   γ : Type u_4
                   δ : Type u_5
                   inst✝⁴ : FunLike F α β
                   inst✝³ : HeytingAlgebra α
                   inst✝² : HeytingAlgebra β
                   inst✝¹ : HeytingAlgebra γ
                   inst✝ : HeytingAlgebra δ
                   f : HeytingHom α β
                   f' : α → β
                   h : Eq f' ⇑f
                   ⊢ ∀ (a b : α), Eq (f' (Max.max a b)) (Max.max (f' a) (f' b))
                 -/
  map_sup' := by simpa only [h] using map_sup f
                 /-
                   🎉 no goals
                 -/
                 /-
                   F : Type u_1
                   α : Type u_2
                   β : Type u_3
                   γ : Type u_4
                   δ : Type u_5
                   inst✝⁴ : FunLike F α β
                   inst✝³ : HeytingAlgebra α
                   inst✝² : HeytingAlgebra β
                   inst✝¹ : HeytingAlgebra γ
                   inst✝ : HeytingAlgebra δ
                   f : HeytingHom α β
                   f' : α → β
                   h : Eq f' ⇑f
                   ⊢ ∀ (a b : α), Eq ({ toFun := f', map_sup' := ⋯ }.toFun (Min.min a b)) (Min.mi …
                 -/
  map_inf' := by simpa only [h] using map_inf f
                 /-
                   🎉 no goals
                 -/
                 /-
                   F : Type u_1
                   α : Type u_2
                   β : Type u_3
                   γ : Type u_4
                   δ : Type u_5
                   inst✝⁴ : FunLike F α β
                   inst✝³ : HeytingAlgebra α
                   inst✝² : HeytingAlgebra β
                   inst✝¹ : HeytingAlgebra γ
                   inst✝ : HeytingAlgebra δ
                   f : HeytingHom α β
                   f' : α → β
                   h : Eq f' ⇑f
                   ⊢ Eq ({ toFun := f', map_sup' := ⋯, map_inf' := ⋯ }.toFun Bot.bot) Bot.bot
                 -/
  map_bot' := by simpa only [h] using map_bot f
                 /-
                   🎉 no goals
                 -/
                  /-
                    F : Type u_1
                    α : Type u_2
                    β : Type u_3
                    γ : Type u_4
                    δ : Type u_5
                    inst✝⁴ : FunLike F α β
                    inst✝³ : HeytingAlgebra α
                    inst✝² : HeytingAlgebra β
                    inst✝¹ : HeytingAlgebra γ
                    inst✝ : HeytingAlgebra δ
                    f : HeytingHom α β
                    f' : α → β
                    h : Eq f' ⇑f
                    ⊢ ∀ (a b : α), Eq ({ toFun := f', map_sup' := ⋯, map_inf' := ⋯ }.toFun (HImp.h …
                  -/
  map_himp' := by simpa only [h] using map_himp f
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem coe_copy (f : HeytingHom α β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : HeytingHom α β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- `id` as a `HeytingHom`. -/
protected def id : HeytingHom α α :=
  { BotHom.id _ with
    toLatticeHom := LatticeHom.id _
    map_himp' := fun _ _ => rfl }


@[simp]
theorem coe_id : ⇑(HeytingHom.id α) = id :=
  rfl


@[simp]
theorem id_apply (a : α) : HeytingHom.id α a = a :=
  rfl


instance : Inhabited (HeytingHom α α) :=
  ⟨HeytingHom.id _⟩


instance : PartialOrder (HeytingHom α β) :=
  PartialOrder.lift _ DFunLike.coe_injective


/-- Composition of `HeytingHom`s as a `HeytingHom`. -/
def comp (f : HeytingHom β γ) (g : HeytingHom α β) : HeytingHom α γ :=
  { f.toLatticeHom.comp g.toLatticeHom with
    toFun := f ∘ g
                   /-
                     F : Type u_1
                     α : Type u_2
                     β : Type u_3
                     γ : Type u_4
                     δ : Type u_5
                     inst✝⁴ : FunLike F α β
                     inst✝³ : HeytingAlgebra α
                     inst✝² : HeytingAlgebra β
                     inst✝¹ : HeytingAlgebra γ
                     inst✝ : HeytingAlgebra δ
                     f : HeytingHom β γ
                     g : HeytingHom α β
                     ⊢ Eq ({ toFun := Function.comp ⇑f ⇑g, map_sup' := ⋯, map_inf' := ⋯ }.toFun Bot …
                   -/
    map_bot' := by simp
                   /-
                     🎉 no goals
                   -/
                               /-
                                 F : Type u_1
                                 α : Type u_2
                                 β : Type u_3
                                 γ : Type u_4
                                 δ : Type u_5
                                 inst✝⁴ : FunLike F α β
                                 inst✝³ : HeytingAlgebra α
                                 inst✝² : HeytingAlgebra β
                                 inst✝¹ : HeytingAlgebra γ
                                 inst✝ : HeytingAlgebra δ
                                 f : HeytingHom β γ
                                 g : HeytingHom α β
                                 a b : α
                                 ⊢ Eq ({ toFun := Function.comp ⇑f ⇑g, map_sup' := ⋯, map_inf' := ⋯ }.toFun (HI …
                               -/
    map_himp' := fun a b => by simp }
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem coe_comp (f : HeytingHom β γ) (g : HeytingHom α β) : ⇑(f.comp g) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : HeytingHom β γ) (g : HeytingHom α β) (a : α) : f.comp g a = f (g a) :=
  rfl


@[simp]
theorem comp_assoc (f : HeytingHom γ δ) (g : HeytingHom β γ) (h : HeytingHom α β) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp]
theorem comp_id (f : HeytingHom α β) : f.comp (HeytingHom.id α) = f :=
  ext fun _ => rfl


@[simp]
theorem id_comp (f : HeytingHom α β) : (HeytingHom.id β).comp f = f :=
  ext fun _ => rfl


@[simp]
theorem cancel_right (hf : Surjective f) : g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => ext <| hf.forall.2 <| DFunLike.ext_iff.1 h, congr_arg (fun a ↦ comp a f)⟩


@[simp]
theorem cancel_left (hg : Injective g) : g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                             /-
                                               α : Type u_2
                                               β : Type u_3
                                               γ : Type u_4
                                               inst✝² : HeytingAlgebra α
                                               inst✝¹ : HeytingAlgebra β
                                               inst✝ : HeytingAlgebra γ
                                               f₁ f₂ : HeytingHom α β
                                               g : HeytingHom β γ
                                               hg : Function.Injective ⇑g
                                               h : Eq (g.comp f₁) (g.comp f₂)
                                               a : α
                                               ⊢ Eq (g (f₁ a)) (g (f₂ a))
                                             -/
  ⟨fun h => HeytingHom.ext fun a => hg <| by rw [← comp_apply, h, comp_apply], congr_arg _⟩
                                             /-
                                               🎉 no goals
                                             -/


instance : FunLike (CoheytingHom α β) α β where
  coe f := f.toFun
                             /-
                               F : Type u_1
                               α : Type u_2
                               β : Type u_3
                               γ : Type u_4
                               δ : Type u_5
                               inst✝⁴ : FunLike F α β
                               inst✝³ : CoheytingAlgebra α
                               inst✝² : CoheytingAlgebra β
                               inst✝¹ : CoheytingAlgebra γ
                               inst✝ : CoheytingAlgebra δ
                               f g : CoheytingHom α β
                               h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by obtain ⟨⟨⟨_, _⟩, _⟩, _⟩ := f; obtain ⟨⟨⟨_, _⟩, _⟩, _⟩ := g; congr
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


instance : CoheytingHomClass (CoheytingHom α β) α β where
  map_sup f := f.map_sup'
  map_inf f := f.map_inf'
  map_top f := f.map_top'
  map_sdiff := CoheytingHom.map_sdiff'

-- @[simp] -- Porting note: not in simp-nf, simp can simplify lhs. Added aux simp lemma

theorem toFun_eq_coe {f : CoheytingHom α β} : f.toFun = (f : α → β) :=
  rfl


@[simp]
theorem toFun_eq_coe_aux {f : CoheytingHom α β} : (↑f.toLatticeHom) = ⇑f :=
  rfl


@[ext]
theorem ext {f g : CoheytingHom α β} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


/-- Copy of a `CoheytingHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (f : CoheytingHom α β) (f' : α → β) (h : f' = f) : CoheytingHom α β where
  toFun := f'
                 /-
                   F : Type u_1
                   α : Type u_2
                   β : Type u_3
                   γ : Type u_4
                   δ : Type u_5
                   inst✝⁴ : FunLike F α β
                   inst✝³ : CoheytingAlgebra α
                   inst✝² : CoheytingAlgebra β
                   inst✝¹ : CoheytingAlgebra γ
                   inst✝ : CoheytingAlgebra δ
                   f : CoheytingHom α β
                   f' : α → β
                   h : Eq f' ⇑f
                   ⊢ ∀ (a b : α), Eq (f' (Max.max a b)) (Max.max (f' a) (f' b))
                 -/
  map_sup' := by simpa only [h] using map_sup f
                 /-
                   🎉 no goals
                 -/
                 /-
                   F : Type u_1
                   α : Type u_2
                   β : Type u_3
                   γ : Type u_4
                   δ : Type u_5
                   inst✝⁴ : FunLike F α β
                   inst✝³ : CoheytingAlgebra α
                   inst✝² : CoheytingAlgebra β
                   inst✝¹ : CoheytingAlgebra γ
                   inst✝ : CoheytingAlgebra δ
                   f : CoheytingHom α β
                   f' : α → β
                   h : Eq f' ⇑f
                   ⊢ ∀ (a b : α), Eq ({ toFun := f', map_sup' := ⋯ }.toFun (Min.min a b)) (Min.mi …
                 -/
  map_inf' := by simpa only [h] using map_inf f
                 /-
                   🎉 no goals
                 -/
                 /-
                   F : Type u_1
                   α : Type u_2
                   β : Type u_3
                   γ : Type u_4
                   δ : Type u_5
                   inst✝⁴ : FunLike F α β
                   inst✝³ : CoheytingAlgebra α
                   inst✝² : CoheytingAlgebra β
                   inst✝¹ : CoheytingAlgebra γ
                   inst✝ : CoheytingAlgebra δ
                   f : CoheytingHom α β
                   f' : α → β
                   h : Eq f' ⇑f
                   ⊢ Eq ({ toFun := f', map_sup' := ⋯, map_inf' := ⋯ }.toFun Top.top) Top.top
                 -/
  map_top' := by simpa only [h] using map_top f
                 /-
                   🎉 no goals
                 -/
                   /-
                     F : Type u_1
                     α : Type u_2
                     β : Type u_3
                     γ : Type u_4
                     δ : Type u_5
                     inst✝⁴ : FunLike F α β
                     inst✝³ : CoheytingAlgebra α
                     inst✝² : CoheytingAlgebra β
                     inst✝¹ : CoheytingAlgebra γ
                     inst✝ : CoheytingAlgebra δ
                     f : CoheytingHom α β
                     f' : α → β
                     h : Eq f' ⇑f
                     ⊢ ∀ (a b : α), Eq ({ toFun := f', map_sup' := ⋯, map_inf' := ⋯ }.toFun (SDiff. …
                   -/
  map_sdiff' := by simpa only [h] using map_sdiff f
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem coe_copy (f : CoheytingHom α β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : CoheytingHom α β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- `id` as a `CoheytingHom`. -/
protected def id : CoheytingHom α α :=
  { TopHom.id _ with
    toLatticeHom := LatticeHom.id _
    map_sdiff' := fun _ _ => rfl }


@[simp]
theorem coe_id : ⇑(CoheytingHom.id α) = id :=
  rfl


@[simp]
theorem id_apply (a : α) : CoheytingHom.id α a = a :=
  rfl


instance : Inhabited (CoheytingHom α α) :=
  ⟨CoheytingHom.id _⟩


instance : PartialOrder (CoheytingHom α β) :=
  PartialOrder.lift _ DFunLike.coe_injective


/-- Composition of `CoheytingHom`s as a `CoheytingHom`. -/
def comp (f : CoheytingHom β γ) (g : CoheytingHom α β) : CoheytingHom α γ :=
  { f.toLatticeHom.comp g.toLatticeHom with
    toFun := f ∘ g
                   /-
                     F : Type u_1
                     α : Type u_2
                     β : Type u_3
                     γ : Type u_4
                     δ : Type u_5
                     inst✝⁴ : FunLike F α β
                     inst✝³ : CoheytingAlgebra α
                     inst✝² : CoheytingAlgebra β
                     inst✝¹ : CoheytingAlgebra γ
                     inst✝ : CoheytingAlgebra δ
                     f : CoheytingHom β γ
                     g : CoheytingHom α β
                     ⊢ Eq ({ toFun := Function.comp ⇑f ⇑g, map_sup' := ⋯, map_inf' := ⋯ }.toFun Top …
                   -/
    map_top' := by simp
                   /-
                     🎉 no goals
                   -/
                                /-
                                  F : Type u_1
                                  α : Type u_2
                                  β : Type u_3
                                  γ : Type u_4
                                  δ : Type u_5
                                  inst✝⁴ : FunLike F α β
                                  inst✝³ : CoheytingAlgebra α
                                  inst✝² : CoheytingAlgebra β
                                  inst✝¹ : CoheytingAlgebra γ
                                  inst✝ : CoheytingAlgebra δ
                                  f : CoheytingHom β γ
                                  g : CoheytingHom α β
                                  a b : α
                                  ⊢ Eq ({ toFun := Function.comp ⇑f ⇑g, map_sup' := ⋯, map_inf' := ⋯ }.toFun (SD …
                                -/
    map_sdiff' := fun a b => by simp }
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem coe_comp (f : CoheytingHom β γ) (g : CoheytingHom α β) : ⇑(f.comp g) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : CoheytingHom β γ) (g : CoheytingHom α β) (a : α) : f.comp g a = f (g a) :=
  rfl


@[simp]
theorem comp_assoc (f : CoheytingHom γ δ) (g : CoheytingHom β γ) (h : CoheytingHom α β) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp]
theorem comp_id (f : CoheytingHom α β) : f.comp (CoheytingHom.id α) = f :=
  ext fun _ => rfl


@[simp]
theorem id_comp (f : CoheytingHom α β) : (CoheytingHom.id β).comp f = f :=
  ext fun _ => rfl


@[simp]
theorem cancel_left (hg : Injective g) : g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                               /-
                                                 α : Type u_2
                                                 β : Type u_3
                                                 γ : Type u_4
                                                 inst✝² : CoheytingAlgebra α
                                                 inst✝¹ : CoheytingAlgebra β
                                                 inst✝ : CoheytingAlgebra γ
                                                 f₁ f₂ : CoheytingHom α β
                                                 g : CoheytingHom β γ
                                                 hg : Function.Injective ⇑g
                                                 h : Eq (g.comp f₁) (g.comp f₂)
                                                 a : α
                                                 ⊢ Eq (g (f₁ a)) (g (f₂ a))
                                               -/
  ⟨fun h => CoheytingHom.ext fun a => hg <| by rw [← comp_apply, h, comp_apply], congr_arg _⟩
                                               /-
                                                 🎉 no goals
                                               -/


instance : FunLike (BiheytingHom α β) α β where
  coe f := f.toFun
                             /-
                               F : Type u_1
                               α : Type u_2
                               β : Type u_3
                               γ : Type u_4
                               δ : Type u_5
                               inst✝⁴ : FunLike F α β
                               inst✝³ : BiheytingAlgebra α
                               inst✝² : BiheytingAlgebra β
                               inst✝¹ : BiheytingAlgebra γ
                               inst✝ : BiheytingAlgebra δ
                               f g : BiheytingHom α β
                               h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by obtain ⟨⟨⟨_, _⟩, _⟩, _⟩ := f; obtain ⟨⟨⟨_, _⟩, _⟩, _⟩ := g; congr
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


instance : BiheytingHomClass (BiheytingHom α β) α β where
  map_sup f := f.map_sup'
  map_inf f := f.map_inf'
  map_himp f := f.map_himp'
  map_sdiff f := f.map_sdiff'

-- @[simp] -- Porting note: not in simp-nf, simp can simplify lhs. Added aux simp lemma

theorem toFun_eq_coe {f : BiheytingHom α β} : f.toFun = (f : α → β) :=
  rfl


@[simp]
theorem toFun_eq_coe_aux {f : BiheytingHom α β} : (↑f.toLatticeHom) = ⇑f :=
  rfl


@[ext]
theorem ext {f g : BiheytingHom α β} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


/-- Copy of a `BiheytingHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (f : BiheytingHom α β) (f' : α → β) (h : f' = f) : BiheytingHom α β where
  toFun := f'
                 /-
                   F : Type u_1
                   α : Type u_2
                   β : Type u_3
                   γ : Type u_4
                   δ : Type u_5
                   inst✝⁴ : FunLike F α β
                   inst✝³ : BiheytingAlgebra α
                   inst✝² : BiheytingAlgebra β
                   inst✝¹ : BiheytingAlgebra γ
                   inst✝ : BiheytingAlgebra δ
                   f : BiheytingHom α β
                   f' : α → β
                   h : Eq f' ⇑f
                   ⊢ ∀ (a b : α), Eq (f' (Max.max a b)) (Max.max (f' a) (f' b))
                 -/
  map_sup' := by simpa only [h] using map_sup f
                 /-
                   🎉 no goals
                 -/
                 /-
                   F : Type u_1
                   α : Type u_2
                   β : Type u_3
                   γ : Type u_4
                   δ : Type u_5
                   inst✝⁴ : FunLike F α β
                   inst✝³ : BiheytingAlgebra α
                   inst✝² : BiheytingAlgebra β
                   inst✝¹ : BiheytingAlgebra γ
                   inst✝ : BiheytingAlgebra δ
                   f : BiheytingHom α β
                   f' : α → β
                   h : Eq f' ⇑f
                   ⊢ ∀ (a b : α), Eq ({ toFun := f', map_sup' := ⋯ }.toFun (Min.min a b)) (Min.mi …
                 -/
  map_inf' := by simpa only [h] using map_inf f
                 /-
                   🎉 no goals
                 -/
                  /-
                    F : Type u_1
                    α : Type u_2
                    β : Type u_3
                    γ : Type u_4
                    δ : Type u_5
                    inst✝⁴ : FunLike F α β
                    inst✝³ : BiheytingAlgebra α
                    inst✝² : BiheytingAlgebra β
                    inst✝¹ : BiheytingAlgebra γ
                    inst✝ : BiheytingAlgebra δ
                    f : BiheytingHom α β
                    f' : α → β
                    h : Eq f' ⇑f
                    ⊢ ∀ (a b : α), Eq ({ toFun := f', map_sup' := ⋯, map_inf' := ⋯ }.toFun (HImp.h …
                  -/
  map_himp' := by simpa only [h] using map_himp f
                  /-
                    🎉 no goals
                  -/
                   /-
                     F : Type u_1
                     α : Type u_2
                     β : Type u_3
                     γ : Type u_4
                     δ : Type u_5
                     inst✝⁴ : FunLike F α β
                     inst✝³ : BiheytingAlgebra α
                     inst✝² : BiheytingAlgebra β
                     inst✝¹ : BiheytingAlgebra γ
                     inst✝ : BiheytingAlgebra δ
                     f : BiheytingHom α β
                     f' : α → β
                     h : Eq f' ⇑f
                     ⊢ ∀ (a b : α), Eq ({ toFun := f', map_sup' := ⋯, map_inf' := ⋯ }.toFun (SDiff. …
                   -/
  map_sdiff' := by simpa only [h] using map_sdiff f
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem coe_copy (f : BiheytingHom α β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : BiheytingHom α β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- `id` as a `BiheytingHom`. -/
protected def id : BiheytingHom α α :=
  { HeytingHom.id _, CoheytingHom.id _ with toLatticeHom := LatticeHom.id _ }


@[simp]
theorem coe_id : ⇑(BiheytingHom.id α) = id :=
  rfl


@[simp]
theorem id_apply (a : α) : BiheytingHom.id α a = a :=
  rfl


instance : Inhabited (BiheytingHom α α) :=
  ⟨BiheytingHom.id _⟩


instance : PartialOrder (BiheytingHom α β) :=
  PartialOrder.lift _ DFunLike.coe_injective


/-- Composition of `BiheytingHom`s as a `BiheytingHom`. -/
def comp (f : BiheytingHom β γ) (g : BiheytingHom α β) : BiheytingHom α γ :=
  { f.toLatticeHom.comp g.toLatticeHom with
    toFun := f ∘ g
                               /-
                                 F : Type u_1
                                 α : Type u_2
                                 β : Type u_3
                                 γ : Type u_4
                                 δ : Type u_5
                                 inst✝⁴ : FunLike F α β
                                 inst✝³ : BiheytingAlgebra α
                                 inst✝² : BiheytingAlgebra β
                                 inst✝¹ : BiheytingAlgebra γ
                                 inst✝ : BiheytingAlgebra δ
                                 f : BiheytingHom β γ
                                 g : BiheytingHom α β
                                 a b : α
                                 ⊢ Eq ({ toFun := Function.comp ⇑f ⇑g, map_sup' := ⋯, map_inf' := ⋯ }.toFun (HI …
                               -/
    map_himp' := fun a b => by simp
                               /-
                                 🎉 no goals
                               -/
                                /-
                                  F : Type u_1
                                  α : Type u_2
                                  β : Type u_3
                                  γ : Type u_4
                                  δ : Type u_5
                                  inst✝⁴ : FunLike F α β
                                  inst✝³ : BiheytingAlgebra α
                                  inst✝² : BiheytingAlgebra β
                                  inst✝¹ : BiheytingAlgebra γ
                                  inst✝ : BiheytingAlgebra δ
                                  f : BiheytingHom β γ
                                  g : BiheytingHom α β
                                  a b : α
                                  ⊢ Eq ({ toFun := Function.comp ⇑f ⇑g, map_sup' := ⋯, map_inf' := ⋯ }.toFun (SD …
                                -/
    map_sdiff' := fun a b => by simp }
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem coe_comp (f : BiheytingHom β γ) (g : BiheytingHom α β) : ⇑(f.comp g) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : BiheytingHom β γ) (g : BiheytingHom α β) (a : α) : f.comp g a = f (g a) :=
  rfl


@[simp]
theorem comp_assoc (f : BiheytingHom γ δ) (g : BiheytingHom β γ) (h : BiheytingHom α β) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp]
theorem comp_id (f : BiheytingHom α β) : f.comp (BiheytingHom.id α) = f :=
  ext fun _ => rfl


@[simp]
theorem id_comp (f : BiheytingHom α β) : (BiheytingHom.id β).comp f = f :=
  ext fun _ => rfl


@[simp]
theorem cancel_left (hg : Injective g) : g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                               /-
                                                 α : Type u_2
                                                 β : Type u_3
                                                 γ : Type u_4
                                                 inst✝² : BiheytingAlgebra α
                                                 inst✝¹ : BiheytingAlgebra β
                                                 inst✝ : BiheytingAlgebra γ
                                                 f₁ f₂ : BiheytingHom α β
                                                 g : BiheytingHom β γ
                                                 hg : Function.Injective ⇑g
                                                 h : Eq (g.comp f₁) (g.comp f₂)
                                                 a : α
                                                 ⊢ Eq (g (f₁ a)) (g (f₂ a))
                                               -/
  ⟨fun h => BiheytingHom.ext fun a => hg <| by rw [← comp_apply, h, comp_apply], congr_arg _⟩
                                               /-
                                                 🎉 no goals
                                               -/


