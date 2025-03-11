/-- The type of `⊔`-preserving functions from `α` to `β`. -/
structure SupHom (α β : Type*) [Max α] [Max β] where
  /-- The underlying function of a `SupHom` -/
  toFun : α → β
  /-- A `SupHom` preserves suprema. -/
  map_sup' (a b : α) : toFun (a ⊔ b) = toFun a ⊔ toFun b


/-- The type of `⊓`-preserving functions from `α` to `β`. -/
structure InfHom (α β : Type*) [Min α] [Min β] where
  /-- The underlying function of an `InfHom` -/
  toFun : α → β
  /-- An `InfHom` preserves infima. -/
  map_inf' (a b : α) : toFun (a ⊓ b) = toFun a ⊓ toFun b


/-- The type of finitary supremum-preserving homomorphisms from `α` to `β`. -/
structure SupBotHom (α β : Type*) [Max α] [Max β] [Bot α] [Bot β] extends SupHom α β where
  /-- A `SupBotHom` preserves the bottom element. -/
  map_bot' : toFun ⊥ = ⊥


/-- The type of finitary infimum-preserving homomorphisms from `α` to `β`. -/
structure InfTopHom (α β : Type*) [Min α] [Min β] [Top α] [Top β] extends InfHom α β where
  /-- An `InfTopHom` preserves the top element. -/
  map_top' : toFun ⊤ = ⊤


/-- The type of lattice homomorphisms from `α` to `β`. -/
structure LatticeHom (α β : Type*) [Lattice α] [Lattice β] extends SupHom α β where
  /-- A `LatticeHom` preserves infima. -/
  map_inf' (a b : α) : toFun (a ⊓ b) = toFun a ⊓ toFun b


/-- The type of bounded lattice homomorphisms from `α` to `β`. -/
structure BoundedLatticeHom (α β : Type*) [Lattice α] [Lattice β] [BoundedOrder α]
  [BoundedOrder β] extends LatticeHom α β where
  /-- A `BoundedLatticeHom` preserves the top element. -/
  map_top' : toFun ⊤ = ⊤
  /-- A `BoundedLatticeHom` preserves the bottom element. -/
  map_bot' : toFun ⊥ = ⊥

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: remove this configuration and use the default configuration.
-- We keep this to be consistent with Lean 3.

/-- `SupHomClass F α β` states that `F` is a type of `⊔`-preserving morphisms.

You should extend this class when you extend `SupHom`. -/
class SupHomClass (F α β : Type*) [Max α] [Max β] [FunLike F α β] : Prop where
  /-- A `SupHomClass` morphism preserves suprema. -/
  map_sup (f : F) (a b : α) : f (a ⊔ b) = f a ⊔ f b


/-- `InfHomClass F α β` states that `F` is a type of `⊓`-preserving morphisms.

You should extend this class when you extend `InfHom`. -/
class InfHomClass (F α β : Type*) [Min α] [Min β] [FunLike F α β] : Prop where
  /-- An `InfHomClass` morphism preserves infima. -/
  map_inf (f : F) (a b : α) : f (a ⊓ b) = f a ⊓ f b


/-- `SupBotHomClass F α β` states that `F` is a type of finitary supremum-preserving morphisms.

You should extend this class when you extend `SupBotHom`. -/
class SupBotHomClass (F α β : Type*) [Max α] [Max β] [Bot α] [Bot β] [FunLike F α β]
  extends SupHomClass F α β : Prop where
  /-- A `SupBotHomClass` morphism preserves the bottom element. -/
  map_bot (f : F) : f ⊥ = ⊥


/-- `InfTopHomClass F α β` states that `F` is a type of finitary infimum-preserving morphisms.

You should extend this class when you extend `SupBotHom`. -/
class InfTopHomClass (F α β : Type*) [Min α] [Min β] [Top α] [Top β] [FunLike F α β]
  extends InfHomClass F α β : Prop where
  /-- An `InfTopHomClass` morphism preserves the top element. -/
  map_top (f : F) : f ⊤ = ⊤


/-- `LatticeHomClass F α β` states that `F` is a type of lattice morphisms.

You should extend this class when you extend `LatticeHom`. -/
class LatticeHomClass (F α β : Type*) [Lattice α] [Lattice β] [FunLike F α β]
  extends SupHomClass F α β : Prop where
  /-- A `LatticeHomClass` morphism preserves infima. -/
  map_inf (f : F) (a b : α) : f (a ⊓ b) = f a ⊓ f b


/-- `BoundedLatticeHomClass F α β` states that `F` is a type of bounded lattice morphisms.

You should extend this class when you extend `BoundedLatticeHom`. -/
class BoundedLatticeHomClass (F α β : Type*) [Lattice α] [Lattice β] [BoundedOrder α]
  [BoundedOrder β] [FunLike F α β] extends LatticeHomClass F α β : Prop where
  /-- A `BoundedLatticeHomClass` morphism preserves the top element. -/
  map_top (f : F) : f ⊤ = ⊤
  /-- A `BoundedLatticeHomClass` morphism preserves the bottom element. -/
  map_bot (f : F) : f ⊥ = ⊥


instance (priority := 100) SupHomClass.toOrderHomClass [SemilatticeSup α] [SemilatticeSup β]
    [SupHomClass F α β] : OrderHomClass F α β :=
  { ‹SupHomClass F α β› with
                                 /-
                                   F : Type u_1
                                   ι : Type u_2
                                   α : Type u_3
                                   β : Type u_4
                                   γ : Type u_5
                                   δ : Type u_6
                                   inst✝³ : FunLike F α β
                                   inst✝² : SemilatticeSup α
                                   inst✝¹ : SemilatticeSup β
                                   inst✝ : SupHomClass F α β
                                   f : F
                                   a b : α
                                   h : LE.le a b
                                   ⊢ LE.le (f a) (f b)
                                 -/
    map_rel := fun f a b h => by rw [← sup_eq_right, ← map_sup, sup_eq_right.2 h] }
                                 /-
                                   🎉 no goals
                                 -/

-- See note [lower instance priority]

instance (priority := 100) InfHomClass.toOrderHomClass [SemilatticeInf α] [SemilatticeInf β]
    [InfHomClass F α β] : OrderHomClass F α β :=
  { ‹InfHomClass F α β› with
                                 /-
                                   F : Type u_1
                                   ι : Type u_2
                                   α : Type u_3
                                   β : Type u_4
                                   γ : Type u_5
                                   δ : Type u_6
                                   inst✝³ : FunLike F α β
                                   inst✝² : SemilatticeInf α
                                   inst✝¹ : SemilatticeInf β
                                   inst✝ : InfHomClass F α β
                                   f : F
                                   a b : α
                                   h : LE.le a b
                                   ⊢ LE.le (f a) (f b)
                                 -/
    map_rel := fun f a b h => by rw [← inf_eq_left, ← map_inf, inf_eq_left.2 h] }
                                 /-
                                   🎉 no goals
                                 -/

-- See note [lower instance priority]

instance (priority := 100) SupBotHomClass.toBotHomClass [Max α] [Max β] [Bot α]
    [Bot β] [SupBotHomClass F α β] : BotHomClass F α β :=
  { ‹SupBotHomClass F α β› with }

-- See note [lower instance priority]

instance (priority := 100) InfTopHomClass.toTopHomClass [Min α] [Min β] [Top α]
    [Top β] [InfTopHomClass F α β] : TopHomClass F α β :=
  { ‹InfTopHomClass F α β› with }

-- See note [lower instance priority]

instance (priority := 100) LatticeHomClass.toInfHomClass [Lattice α] [Lattice β]
    [LatticeHomClass F α β] : InfHomClass F α β :=
  { ‹LatticeHomClass F α β› with }

-- See note [lower instance priority]

instance (priority := 100) BoundedLatticeHomClass.toSupBotHomClass [Lattice α] [Lattice β]
    [BoundedOrder α] [BoundedOrder β] [BoundedLatticeHomClass F α β] :
    SupBotHomClass F α β :=
  { ‹BoundedLatticeHomClass F α β› with }

-- See note [lower instance priority]

instance (priority := 100) BoundedLatticeHomClass.toInfTopHomClass [Lattice α] [Lattice β]
    [BoundedOrder α] [BoundedOrder β] [BoundedLatticeHomClass F α β] :
    InfTopHomClass F α β :=
  { ‹BoundedLatticeHomClass F α β› with }

-- See note [lower instance priority]

instance (priority := 100) BoundedLatticeHomClass.toBoundedOrderHomClass [Lattice α]
    [Lattice β] [BoundedOrder α] [BoundedOrder β] [BoundedLatticeHomClass F α β] :
    BoundedOrderHomClass F α β :=
{ show OrderHomClass F α β from inferInstance, ‹BoundedLatticeHomClass F α β› with }


instance (priority := 100) OrderIsoClass.toSupHomClass [SemilatticeSup α] [SemilatticeSup β]
    [OrderIsoClass F α β] : SupHomClass F α β :=
  { show OrderHomClass F α β from inferInstance with
    map_sup := fun f a b =>
                                      /-
                                        F : Type u_1
                                        ι : Type u_2
                                        α : Type u_3
                                        β : Type u_4
                                        γ : Type u_5
                                        δ : Type u_6
                                        inst✝³ : EquivLike F α β
                                        inst✝² : SemilatticeSup α
                                        inst✝¹ : SemilatticeSup β
                                        inst✝ : OrderIsoClass F α β
                                        f : F
                                        a b : α
                                        c : β
                                        ⊢ Iff (LE.le (f (Max.max a b)) c) (LE.le (Max.max (f a) (f b)) c)
                                      -/
      eq_of_forall_ge_iff fun c => by simp only [← le_map_inv_iff, sup_le_iff] }
                                      /-
                                        🎉 no goals
                                      -/


-- See note [lower instance priority]

instance (priority := 100) OrderIsoClass.toInfHomClass [SemilatticeInf α] [SemilatticeInf β]
    [OrderIsoClass F α β] : InfHomClass F α β :=
  { show OrderHomClass F α β from inferInstance with
    map_inf := fun f a b =>
                                      /-
                                        F : Type u_1
                                        ι : Type u_2
                                        α : Type u_3
                                        β : Type u_4
                                        γ : Type u_5
                                        δ : Type u_6
                                        inst✝³ : EquivLike F α β
                                        inst✝² : SemilatticeInf α
                                        inst✝¹ : SemilatticeInf β
                                        inst✝ : OrderIsoClass F α β
                                        f : F
                                        a b : α
                                        c : β
                                        ⊢ Iff (LE.le c (f (Min.min a b))) (LE.le c (Min.min (f a) (f b)))
                                      -/
      eq_of_forall_le_iff fun c => by simp only [← map_inv_le_iff, le_inf_iff] }
                                      /-
                                        🎉 no goals
                                      -/

-- See note [lower instance priority]

instance (priority := 100) OrderIsoClass.toSupBotHomClass [SemilatticeSup α] [OrderBot α]
    [SemilatticeSup β] [OrderBot β] [OrderIsoClass F α β] : SupBotHomClass F α β :=
  { OrderIsoClass.toSupHomClass, OrderIsoClass.toBotHomClass with }

-- See note [lower instance priority]

instance (priority := 100) OrderIsoClass.toInfTopHomClass [SemilatticeInf α] [OrderTop α]
    [SemilatticeInf β] [OrderTop β] [OrderIsoClass F α β] : InfTopHomClass F α β :=
  { OrderIsoClass.toInfHomClass, OrderIsoClass.toTopHomClass with }

-- See note [lower instance priority]

instance (priority := 100) OrderIsoClass.toLatticeHomClass [Lattice α] [Lattice β]
    [OrderIsoClass F α β] : LatticeHomClass F α β :=
  { OrderIsoClass.toSupHomClass, OrderIsoClass.toInfHomClass with }

-- See note [lower instance priority]

instance (priority := 100) OrderIsoClass.toBoundedLatticeHomClass [Lattice α] [Lattice β]
    [BoundedOrder α] [BoundedOrder β] [OrderIsoClass F α β] :
    BoundedLatticeHomClass F α β :=
  { OrderIsoClass.toLatticeHomClass, OrderIsoClass.toBoundedOrderHomClass with }


/-- We can regard an injective map preserving binary infima as an order embedding. -/
@[simps! apply]
def orderEmbeddingOfInjective [SemilatticeInf α] [SemilatticeInf β] (f : F) [InfHomClass F α β]
    (hf : Injective f) : α ↪o β :=
  OrderEmbedding.ofMapLEIff f (fun x y ↦ by
    /-
      F : Type u_1
      ι : Type u_2
      α : Type u_3
      β : Type u_4
      γ : Type u_5
      δ : Type u_6
      inst✝³ : FunLike F α β
      inst✝² : SemilatticeInf α
      inst✝¹ : SemilatticeInf β
      f : F
      inst✝ : InfHomClass F α β
      hf : Function.Injective ⇑f
      x y : α
      ⊢ Iff (LE.le (f x) (f y)) (LE.le x y)
    -/
    refine ⟨fun h ↦ ?_, fun h ↦ OrderHomClass.mono f h⟩
    /-
      F : Type u_1
      ι : Type u_2
      α : Type u_3
      β : Type u_4
      γ : Type u_5
      δ : Type u_6
      inst✝³ : FunLike F α β
      inst✝² : SemilatticeInf α
      inst✝¹ : SemilatticeInf β
      f : F
      inst✝ : InfHomClass F α β
      hf : Function.Injective ⇑f
      x y : α
      h : LE.le (f x) (f y)
      ⊢ LE.le x y
    -/
    rwa [← inf_eq_left, ← hf.eq_iff, map_inf, inf_eq_left])
    /-
      🎉 no goals
    -/


theorem Disjoint.map (h : Disjoint a b) : Disjoint (f a) (f b) := by
  /-
    F : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝⁵ : Lattice α
    inst✝⁴ : BoundedOrder α
    inst✝³ : Lattice β
    inst✝² : BoundedOrder β
    inst✝¹ : FunLike F α β
    inst✝ : BoundedLatticeHomClass F α β
    f : F
    a b : α
    h : Disjoint a b
    ⊢ Disjoint (f a) (f b)
  -/
  rw [disjoint_iff, ← map_inf, h.eq_bot, map_bot]
  /-
    🎉 no goals
  -/


theorem Codisjoint.map (h : Codisjoint a b) : Codisjoint (f a) (f b) := by
  /-
    F : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝⁵ : Lattice α
    inst✝⁴ : BoundedOrder α
    inst✝³ : Lattice β
    inst✝² : BoundedOrder β
    inst✝¹ : FunLike F α β
    inst✝ : BoundedLatticeHomClass F α β
    f : F
    a b : α
    h : Codisjoint a b
    ⊢ Codisjoint (f a) (f b)
  -/
  rw [codisjoint_iff, ← map_sup, h.eq_top, map_top]
  /-
    🎉 no goals
  -/


theorem IsCompl.map (h : IsCompl a b) : IsCompl (f a) (f b) :=
  ⟨h.1.map _, h.2.map _⟩


/-- Special case of `map_compl` for boolean algebras. -/
theorem map_compl' (a : α) : f aᶜ = (f a)ᶜ :=
  (isCompl_compl.map _).compl_eq.symm


/-- Special case of `map_sdiff` for boolean algebras. -/
theorem map_sdiff' (a b : α) : f (a \ b) = f a \ f b := by
  /-
    F : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝³ : BooleanAlgebra α
    inst✝² : BooleanAlgebra β
    inst✝¹ : FunLike F α β
    inst✝ : BoundedLatticeHomClass F α β
    f : F
    a b : α
    ⊢ Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
  -/
  rw [sdiff_eq, sdiff_eq, map_inf, map_compl']
  /-
    🎉 no goals
  -/


open scoped symmDiff in
/-- Special case of `map_symmDiff` for boolean algebras. -/
theorem map_symmDiff' (a b : α) : f (a ∆ b) = f a ∆ f b := by
  /-
    F : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝³ : BooleanAlgebra α
    inst✝² : BooleanAlgebra β
    inst✝¹ : FunLike F α β
    inst✝ : BoundedLatticeHomClass F α β
    f : F
    a b : α
    ⊢ Eq (f (symmDiff a b)) (symmDiff (f a) (f b))
  -/
  rw [symmDiff, symmDiff, map_sup, map_sdiff', map_sdiff']
  /-
    🎉 no goals
  -/


instance [Max α] [Max β] [SupHomClass F α β] : CoeTC F (SupHom α β) :=
  ⟨fun f => ⟨f, map_sup f⟩⟩


instance [Min α] [Min β] [InfHomClass F α β] : CoeTC F (InfHom α β) :=
  ⟨fun f => ⟨f, map_inf f⟩⟩


instance [Max α] [Max β] [Bot α] [Bot β] [SupBotHomClass F α β] : CoeTC F (SupBotHom α β) :=
  ⟨fun f => ⟨f, map_bot f⟩⟩


instance [Min α] [Min β] [Top α] [Top β] [InfTopHomClass F α β] : CoeTC F (InfTopHom α β) :=
  ⟨fun f => ⟨f, map_top f⟩⟩


instance [Lattice α] [Lattice β] [LatticeHomClass F α β] : CoeTC F (LatticeHom α β) :=
  ⟨fun f =>
    { toFun := f
      map_sup' := map_sup f
      map_inf' := map_inf f }⟩


instance [Lattice α] [Lattice β] [BoundedOrder α] [BoundedOrder β] [BoundedLatticeHomClass F α β] :
    CoeTC F (BoundedLatticeHom α β) :=
  ⟨fun f =>
    { (f : LatticeHom α β) with
      toFun := f
      map_top' := map_top f
      map_bot' := map_bot f }⟩


instance : FunLike (SupHom α β) α β where
  coe := SupHom.toFun
                             /-
                               F : Type u_1
                               ι : Type u_2
                               α : Type u_3
                               β : Type u_4
                               γ : Type u_5
                               δ : Type u_6
                               inst✝⁴ : FunLike F α β
                               inst✝³ : Max α
                               inst✝² : Max β
                               inst✝¹ : Max γ
                               inst✝ : Max δ
                               f g : SupHom α β
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


instance : SupHomClass (SupHom α β) α β where
  map_sup := SupHom.map_sup'


@[simp] lemma toFun_eq_coe (f : SupHom α β) : f.toFun = f := rfl


@[simp, norm_cast] lemma coe_mk (f : α → β) (hf) : ⇑(mk f hf) = f := rfl


@[ext]
theorem ext {f g : SupHom α β} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


/-- Copy of a `SupHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (f : SupHom α β) (f' : α → β) (h : f' = f) : SupHom α β where
  toFun := f'
  map_sup' := h.symm ▸ f.map_sup'


@[simp]
theorem coe_copy (f : SupHom α β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : SupHom α β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- `id` as a `SupHom`. -/
protected def id : SupHom α α :=
  ⟨id, fun _ _ => rfl⟩


instance : Inhabited (SupHom α α) :=
  ⟨SupHom.id α⟩


@[simp]
theorem coe_id : ⇑(SupHom.id α) = id :=
  rfl


@[simp]
theorem id_apply (a : α) : SupHom.id α a = a :=
  rfl


/-- Composition of `SupHom`s as a `SupHom`. -/
def comp (f : SupHom β γ) (g : SupHom α β) : SupHom α γ where
  toFun := f ∘ g
                     /-
                       F : Type u_1
                       ι : Type u_2
                       α : Type u_3
                       β : Type u_4
                       γ : Type u_5
                       δ : Type u_6
                       inst✝⁴ : FunLike F α β
                       inst✝³ : Max α
                       inst✝² : Max β
                       inst✝¹ : Max γ
                       inst✝ : Max δ
                       f : SupHom β γ
                       g : SupHom α β
                       a b : α
                       ⊢ Eq (Function.comp (⇑f) (⇑g) (Max.max a b)) (Max.max (Function.comp (⇑f) (⇑g) …
                     -/
  map_sup' a b := by rw [comp_apply, map_sup, map_sup]; rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem coe_comp (f : SupHom β γ) (g : SupHom α β) : (f.comp g : α → γ) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : SupHom β γ) (g : SupHom α β) (a : α) : (f.comp g) a = f (g a) :=
  rfl


@[simp]
theorem comp_assoc (f : SupHom γ δ) (g : SupHom β γ) (h : SupHom α β) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp] theorem comp_id (f : SupHom α β) : f.comp (SupHom.id α) = f := rfl


@[simp] theorem id_comp (f : SupHom α β) : (SupHom.id β).comp f = f := rfl


@[simp]
theorem cancel_right {g₁ g₂ : SupHom β γ} {f : SupHom α β} (hf : Surjective f) :
    g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => SupHom.ext <| hf.forall.2 <| DFunLike.ext_iff.1 h, fun h => congr_arg₂ _ h rfl⟩


@[simp]
theorem cancel_left {g : SupHom β γ} {f₁ f₂ : SupHom α β} (hg : Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                         /-
                                           α : Type u_3
                                           β : Type u_4
                                           γ : Type u_5
                                           inst✝² : Max α
                                           inst✝¹ : Max β
                                           inst✝ : Max γ
                                           g : SupHom β γ
                                           f₁ f₂ : SupHom α β
                                           hg : Function.Injective ⇑g
                                           h : Eq (g.comp f₁) (g.comp f₂)
                                           a : α
                                           ⊢ Eq (g (f₁ a)) (g (f₂ a))
                                         -/
  ⟨fun h => SupHom.ext fun a => hg <| by rw [← SupHom.comp_apply, h, SupHom.comp_apply],
                                         /-
                                           🎉 no goals
                                         -/
    congr_arg _⟩


/-- The constant function as a `SupHom`. -/
def const (b : β) : SupHom α β := ⟨fun _ ↦ b, fun _ _ ↦ (sup_idem _).symm⟩


@[simp]
theorem coe_const (b : β) : ⇑(const α b) = Function.const α b :=
  rfl


@[simp]
theorem const_apply (b : β) (a : α) : const α b a = b :=
  rfl


instance : Max (SupHom α β) :=
  ⟨fun f g =>
    ⟨f ⊔ g, fun a b => by
      /-
        F : Type u_1
        ι : Type u_2
        α : Type u_3
        β : Type u_4
        γ : Type u_5
        δ : Type u_6
        inst✝² : FunLike F α β
        inst✝¹ : Max α
        inst✝ : SemilatticeSup β
        f g : SupHom α β
        a b : α
        ⊢ Eq (Max.max (⇑f) (⇑g) (Max.max a b)) (Max.max (Max.max (⇑f) (⇑g) a) (Max.max …
      -/
      rw [Pi.sup_apply, map_sup, map_sup]
      /-
        F : Type u_1
        ι : Type u_2
        α : Type u_3
        β : Type u_4
        γ : Type u_5
        δ : Type u_6
        inst✝² : FunLike F α β
        inst✝¹ : Max α
        inst✝ : SemilatticeSup β
        f g : SupHom α β
        a b : α
        ⊢ Eq (Max.max (Max.max (f a) (f b)) (Max.max (g a) (g b))) (Max.max (Max.max ( …
      -/
      exact sup_sup_sup_comm _ _ _ _⟩⟩
      /-
        🎉 no goals
      -/


instance : SemilatticeSup (SupHom α β) :=
  (DFunLike.coe_injective.semilatticeSup _) fun _ _ => rfl


instance [Bot β] : Bot (SupHom α β) :=
  ⟨SupHom.const α ⊥⟩


instance [Top β] : Top (SupHom α β) :=
  ⟨SupHom.const α ⊤⟩


instance [OrderBot β] : OrderBot (SupHom α β) :=
  OrderBot.lift ((↑) : _ → α → β) (fun _ _ => id) rfl


instance [OrderTop β] : OrderTop (SupHom α β) :=
  OrderTop.lift ((↑) : _ → α → β) (fun _ _ => id) rfl


instance [BoundedOrder β] : BoundedOrder (SupHom α β) :=
  BoundedOrder.lift ((↑) : _ → α → β) (fun _ _ => id) rfl rfl


@[simp]
theorem coe_sup (f g : SupHom α β) : DFunLike.coe (f ⊔ g) = f ⊔ g :=
  rfl


@[simp]
theorem coe_bot [Bot β] : ⇑(⊥ : SupHom α β) = ⊥ :=
  rfl


@[simp]
theorem coe_top [Top β] : ⇑(⊤ : SupHom α β) = ⊤ :=
  rfl


@[simp]
theorem sup_apply (f g : SupHom α β) (a : α) : (f ⊔ g) a = f a ⊔ g a :=
  rfl


@[simp]
theorem bot_apply [Bot β] (a : α) : (⊥ : SupHom α β) a = ⊥ :=
  rfl


@[simp]
theorem top_apply [Top β] (a : α) : (⊤ : SupHom α β) a = ⊤ :=
  rfl


/-- `Subtype.val` as a `SupHom`. -/
def subtypeVal {P : β → Prop}
    (Psup : ∀ ⦃x y : β⦄, P x → P y → P (x ⊔ y)) :
    letI := Subtype.semilatticeSup Psup
    SupHom {x : β // P x} β :=
  letI := Subtype.semilatticeSup Psup
                      /-
                        F : Type u_1
                        ι : Type u_2
                        α : Type u_3
                        β : Type u_4
                        γ : Type u_5
                        δ : Type u_6
                        inst✝² : FunLike F α β
                        inst✝¹ : Max α
                        inst✝ : SemilatticeSup β
                        P : β → Prop
                        Psup : ∀ ⦃x y : β⦄, P x → P y → P (Max.max x y)
                        this : SemilatticeSup (Subtype fun x => P x) := Subtype.semilatticeSup Psup
                        ⊢ ∀ (a b : Subtype fun x => P x), Eq (↑(Max.max a b)) (Max.max ↑a ↑b)
                      -/
  .mk Subtype.val (by simp)
                      /-
                        🎉 no goals
                      -/


@[simp]
lemma subtypeVal_apply {P : β → Prop}
    (Psup : ∀ ⦃x y : β⦄, P x → P y → P (x ⊔ y)) (x : {x : β // P x}) :
    subtypeVal Psup x = x := rfl


@[simp]
lemma subtypeVal_coe {P : β → Prop}
    (Psup : ∀ ⦃x y : β⦄, P x → P y → P (x ⊔ y)) :
    ⇑(subtypeVal Psup) = Subtype.val := rfl


instance : FunLike (InfHom α β) α β where
  coe := InfHom.toFun
                             /-
                               F : Type u_1
                               ι : Type u_2
                               α : Type u_3
                               β : Type u_4
                               γ : Type u_5
                               δ : Type u_6
                               inst✝⁴ : FunLike F α β
                               inst✝³ : Min α
                               inst✝² : Min β
                               inst✝¹ : Min γ
                               inst✝ : Min δ
                               f g : InfHom α β
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


instance : InfHomClass (InfHom α β) α β where
  map_inf := InfHom.map_inf'


@[simp] lemma toFun_eq_coe (f : InfHom α β) : f.toFun = (f : α → β) := rfl


@[ext]
theorem ext {f g : InfHom α β} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


/-- Copy of an `InfHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (f : InfHom α β) (f' : α → β) (h : f' = f) : InfHom α β where
  toFun := f'
  map_inf' := h.symm ▸ f.map_inf'


@[simp]
theorem coe_copy (f : InfHom α β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : InfHom α β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- `id` as an `InfHom`. -/
protected def id : InfHom α α :=
  ⟨id, fun _ _ => rfl⟩


instance : Inhabited (InfHom α α) :=
  ⟨InfHom.id α⟩


@[simp]
theorem coe_id : ⇑(InfHom.id α) = id :=
  rfl


@[simp]
theorem id_apply (a : α) : InfHom.id α a = a :=
  rfl


/-- Composition of `InfHom`s as an `InfHom`. -/
def comp (f : InfHom β γ) (g : InfHom α β) : InfHom α γ where
  toFun := f ∘ g
                     /-
                       F : Type u_1
                       ι : Type u_2
                       α : Type u_3
                       β : Type u_4
                       γ : Type u_5
                       δ : Type u_6
                       inst✝⁴ : FunLike F α β
                       inst✝³ : Min α
                       inst✝² : Min β
                       inst✝¹ : Min γ
                       inst✝ : Min δ
                       f : InfHom β γ
                       g : InfHom α β
                       a b : α
                       ⊢ Eq (Function.comp (⇑f) (⇑g) (Min.min a b)) (Min.min (Function.comp (⇑f) (⇑g) …
                     -/
  map_inf' a b := by rw [comp_apply, map_inf, map_inf]; rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem coe_comp (f : InfHom β γ) (g : InfHom α β) : (f.comp g : α → γ) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : InfHom β γ) (g : InfHom α β) (a : α) : (f.comp g) a = f (g a) :=
  rfl


@[simp]
theorem comp_assoc (f : InfHom γ δ) (g : InfHom β γ) (h : InfHom α β) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp] theorem comp_id (f : InfHom α β) : f.comp (InfHom.id α) = f := rfl


@[simp] theorem id_comp (f : InfHom α β) : (InfHom.id β).comp f = f := rfl


@[simp]
theorem cancel_right {g₁ g₂ : InfHom β γ} {f : InfHom α β} (hf : Surjective f) :
    g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => InfHom.ext <| hf.forall.2 <| DFunLike.ext_iff.1 h, fun h => congr_arg₂ _ h rfl⟩


@[simp]
theorem cancel_left {g : InfHom β γ} {f₁ f₂ : InfHom α β} (hg : Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                         /-
                                           α : Type u_3
                                           β : Type u_4
                                           γ : Type u_5
                                           inst✝² : Min α
                                           inst✝¹ : Min β
                                           inst✝ : Min γ
                                           g : InfHom β γ
                                           f₁ f₂ : InfHom α β
                                           hg : Function.Injective ⇑g
                                           h : Eq (g.comp f₁) (g.comp f₂)
                                           a : α
                                           ⊢ Eq (g (f₁ a)) (g (f₂ a))
                                         -/
  ⟨fun h => InfHom.ext fun a => hg <| by rw [← InfHom.comp_apply, h, InfHom.comp_apply],
                                         /-
                                           🎉 no goals
                                         -/
    congr_arg _⟩


/-- The constant function as an `InfHom`. -/
def const (b : β) : InfHom α β := ⟨fun _ ↦ b, fun _ _ ↦ (inf_idem _).symm⟩


instance : Min (InfHom α β) :=
  ⟨fun f g =>
    ⟨f ⊓ g, fun a b => by
      /-
        F : Type u_1
        ι : Type u_2
        α : Type u_3
        β : Type u_4
        γ : Type u_5
        δ : Type u_6
        inst✝² : FunLike F α β
        inst✝¹ : Min α
        inst✝ : SemilatticeInf β
        f g : InfHom α β
        a b : α
        ⊢ Eq (Min.min (⇑f) (⇑g) (Min.min a b)) (Min.min (Min.min (⇑f) (⇑g) a) (Min.min …
      -/
      rw [Pi.inf_apply, map_inf, map_inf]
      /-
        F : Type u_1
        ι : Type u_2
        α : Type u_3
        β : Type u_4
        γ : Type u_5
        δ : Type u_6
        inst✝² : FunLike F α β
        inst✝¹ : Min α
        inst✝ : SemilatticeInf β
        f g : InfHom α β
        a b : α
        ⊢ Eq (Min.min (Min.min (f a) (f b)) (Min.min (g a) (g b))) (Min.min (Min.min ( …
      -/
      exact inf_inf_inf_comm _ _ _ _⟩⟩
      /-
        🎉 no goals
      -/


instance : SemilatticeInf (InfHom α β) :=
  (DFunLike.coe_injective.semilatticeInf _) fun _ _ => rfl


instance [Bot β] : Bot (InfHom α β) :=
  ⟨InfHom.const α ⊥⟩


instance [Top β] : Top (InfHom α β) :=
  ⟨InfHom.const α ⊤⟩


instance [OrderBot β] : OrderBot (InfHom α β) :=
  OrderBot.lift ((↑) : _ → α → β) (fun _ _ => id) rfl


instance [OrderTop β] : OrderTop (InfHom α β) :=
  OrderTop.lift ((↑) : _ → α → β) (fun _ _ => id) rfl


instance [BoundedOrder β] : BoundedOrder (InfHom α β) :=
  BoundedOrder.lift ((↑) : _ → α → β) (fun _ _ => id) rfl rfl


@[simp]
theorem coe_inf (f g : InfHom α β) : DFunLike.coe (f ⊓ g) = f ⊓ g :=
  rfl


@[simp]
theorem coe_bot [Bot β] : ⇑(⊥ : InfHom α β) = ⊥ :=
  rfl


@[simp]
theorem coe_top [Top β] : ⇑(⊤ : InfHom α β) = ⊤ :=
  rfl


@[simp]
theorem inf_apply (f g : InfHom α β) (a : α) : (f ⊓ g) a = f a ⊓ g a :=
  rfl


@[simp]
theorem bot_apply [Bot β] (a : α) : (⊥ : InfHom α β) a = ⊥ :=
  rfl


@[simp]
theorem top_apply [Top β] (a : α) : (⊤ : InfHom α β) a = ⊤ :=
  rfl


/-- `Subtype.val` as an `InfHom`. -/
def subtypeVal {P : β → Prop}
    (Pinf : ∀ ⦃x y : β⦄, P x → P y → P (x ⊓ y)) :
    letI := Subtype.semilatticeInf Pinf
    InfHom {x : β // P x} β :=
  letI := Subtype.semilatticeInf Pinf
                      /-
                        F : Type u_1
                        ι : Type u_2
                        α : Type u_3
                        β : Type u_4
                        γ : Type u_5
                        δ : Type u_6
                        inst✝² : FunLike F α β
                        inst✝¹ : Min α
                        inst✝ : SemilatticeInf β
                        P : β → Prop
                        Pinf : ∀ ⦃x y : β⦄, P x → P y → P (Min.min x y)
                        this : SemilatticeInf (Subtype fun x => P x) := Subtype.semilatticeInf Pinf
                        ⊢ ∀ (a b : Subtype fun x => P x), Eq (↑(Min.min a b)) (Min.min ↑a ↑b)
                      -/
  .mk Subtype.val (by simp)
                      /-
                        🎉 no goals
                      -/


@[simp]
lemma subtypeVal_apply {P : β → Prop}
    (Pinf : ∀ ⦃x y : β⦄, P x → P y → P (x ⊓ y)) (x : {x : β // P x}) :
    subtypeVal Pinf x = x := rfl


@[simp]
lemma subtypeVal_coe {P : β → Prop}
    (Pinf : ∀ ⦃x y : β⦄, P x → P y → P (x ⊓ y)) :
    ⇑(subtypeVal Pinf) = Subtype.val := rfl


/-- Reinterpret a `SupBotHom` as a `BotHom`. -/
def toBotHom (f : SupBotHom α β) : BotHom α β :=
  { f with }


instance : FunLike (SupBotHom α β) α β where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      F : Type u_1
      ι : Type u_2
      α : Type u_3
      β : Type u_4
      γ : Type u_5
      δ : Type u_6
      inst✝⁸ : FunLike F α β
      inst✝⁷ : Max α
      inst✝⁶ : Bot α
      inst✝⁵ : Max β
      inst✝⁴ : Bot β
      inst✝³ : Max γ
      inst✝² : Bot γ
      inst✝¹ : Max δ
      inst✝ : Bot δ
      f g : SupBotHom α β
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    obtain ⟨⟨_, _⟩, _⟩ := f
    /-
      case mk.mk
      F : Type u_1
      ι : Type u_2
      α : Type u_3
      β : Type u_4
      γ : Type u_5
      δ : Type u_6
      inst✝⁸ : FunLike F α β
      inst✝⁷ : Max α
      inst✝⁶ : Bot α
      inst✝⁵ : Max β
      inst✝⁴ : Bot β
      inst✝³ : Max γ
      inst✝² : Bot γ
      inst✝¹ : Max δ
      inst✝ : Bot δ
      g : SupBotHom α β
      toFun✝ : α → β
      map_sup'✝ : ∀ (a b : α), Eq (toFun✝ (Max.max a b)) (Max.max (toFun✝ a) (toFun✝ …
      map_bot'✝ : Eq ({ toFun := toFun✝, map_sup' := map_sup'✝ }.toFun Bot.bot) Bot. …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝, map_sup' := map_sup'✝, map_bot'  …
      ⊢ Eq { toFun := toFun✝, map_sup' := map_sup'✝, map_bot' := map_bot'✝ } g
    -/
    obtain ⟨⟨_, _⟩, _⟩ := g
    /-
      case mk.mk.mk.mk
      F : Type u_1
      ι : Type u_2
      α : Type u_3
      β : Type u_4
      γ : Type u_5
      δ : Type u_6
      inst✝⁸ : FunLike F α β
      inst✝⁷ : Max α
      inst✝⁶ : Bot α
      inst✝⁵ : Max β
      inst✝⁴ : Bot β
      inst✝³ : Max γ
      inst✝² : Bot γ
      inst✝¹ : Max δ
      inst✝ : Bot δ
      toFun✝¹ : α → β
      map_sup'✝¹ : ∀ (a b : α), Eq (toFun✝¹ (Max.max a b)) (Max.max (toFun✝¹ a) (toF …
      map_bot'✝¹ : Eq ({ toFun := toFun✝¹, map_sup' := map_sup'✝¹ }.toFun Bot.bot) B …
      toFun✝ : α → β
      map_sup'✝ : ∀ (a b : α), Eq (toFun✝ (Max.max a b)) (Max.max (toFun✝ a) (toFun✝ …
      map_bot'✝ : Eq ({ toFun := toFun✝, map_sup' := map_sup'✝ }.toFun Bot.bot) Bot. …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝¹, map_sup' := map_sup'✝¹, map_bot …
      ⊢ Eq { toFun := toFun✝¹, map_sup' := map_sup'✝¹, map_bot' := map_bot'✝¹ } { to …
    -/
    congr
    /-
      🎉 no goals
    -/


instance : SupBotHomClass (SupBotHom α β) α β where
  map_sup f := f.map_sup'
  map_bot f := f.map_bot'


lemma toFun_eq_coe (f : SupBotHom α β) : f.toFun = f := rfl


@[simp] lemma coe_toSupHom (f : SupBotHom α β) : ⇑f.toSupHom = f := rfl

@[simp] lemma coe_toBotHom (f : SupBotHom α β) : ⇑f.toBotHom = f := rfl

@[simp] lemma coe_mk (f : SupHom α β) (hf) : ⇑(mk f hf) = f := rfl


@[ext]
theorem ext {f g : SupBotHom α β} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


/-- Copy of a `SupBotHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (f : SupBotHom α β) (f' : α → β) (h : f' = f) : SupBotHom α β :=
  { f.toBotHom.copy f' h with toSupHom := f.toSupHom.copy f' h }


@[simp]
theorem coe_copy (f : SupBotHom α β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : SupBotHom α β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- `id` as a `SupBotHom`. -/
@[simps]
protected def id : SupBotHom α α :=
  ⟨SupHom.id α, rfl⟩


instance : Inhabited (SupBotHom α α) :=
  ⟨SupBotHom.id α⟩


@[simp]
theorem coe_id : ⇑(SupBotHom.id α) = id :=
  rfl


@[simp]
theorem id_apply (a : α) : SupBotHom.id α a = a :=
  rfl


/-- Composition of `SupBotHom`s as a `SupBotHom`. -/
def comp (f : SupBotHom β γ) (g : SupBotHom α β) : SupBotHom α γ :=
  { f.toSupHom.comp g.toSupHom, f.toBotHom.comp g.toBotHom with }


@[simp]
theorem coe_comp (f : SupBotHom β γ) (g : SupBotHom α β) : (f.comp g : α → γ) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : SupBotHom β γ) (g : SupBotHom α β) (a : α) : (f.comp g) a = f (g a) :=
  rfl


@[simp]
theorem comp_assoc (f : SupBotHom γ δ) (g : SupBotHom β γ) (h : SupBotHom α β) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp] theorem comp_id (f : SupBotHom α β) : f.comp (SupBotHom.id α) = f := rfl


@[simp] theorem id_comp (f : SupBotHom α β) : (SupBotHom.id β).comp f = f := rfl


@[simp]
theorem cancel_right {g₁ g₂ : SupBotHom β γ} {f : SupBotHom α β} (hf : Surjective f) :
    g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => ext <| hf.forall.2 <| DFunLike.ext_iff.1 h, fun h => congr_arg₂ _ h rfl⟩


@[simp]
theorem cancel_left {g : SupBotHom β γ} {f₁ f₂ : SupBotHom α β} (hg : Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                            /-
                                              α : Type u_3
                                              β : Type u_4
                                              γ : Type u_5
                                              inst✝⁵ : Max α
                                              inst✝⁴ : Bot α
                                              inst✝³ : Max β
                                              inst✝² : Bot β
                                              inst✝¹ : Max γ
                                              inst✝ : Bot γ
                                              g : SupBotHom β γ
                                              f₁ f₂ : SupBotHom α β
                                              hg : Function.Injective ⇑g
                                              h : Eq (g.comp f₁) (g.comp f₂)
                                              a : α
                                              ⊢ Eq (g (f₁ a)) (g (f₂ a))
                                            -/
  ⟨fun h => SupBotHom.ext fun a => hg <| by rw [← comp_apply, h, comp_apply], congr_arg _⟩
                                            /-
                                              🎉 no goals
                                            -/


instance : Max (SupBotHom α β) :=
  ⟨fun f g => { f.toBotHom ⊔ g.toBotHom with toSupHom := f.toSupHom ⊔ g.toSupHom }⟩


instance : SemilatticeSup (SupBotHom α β) :=
  (DFunLike.coe_injective.semilatticeSup _) fun _ _ => rfl


instance : OrderBot (SupBotHom α β) where
  bot := ⟨⊥, rfl⟩
  bot_le _ _ := bot_le


@[simp]
theorem coe_sup (f g : SupBotHom α β) : DFunLike.coe (f ⊔ g) = f ⊔ g :=
  rfl


@[simp]
theorem coe_bot : ⇑(⊥ : SupBotHom α β) = ⊥ :=
  rfl


@[simp]
theorem sup_apply (f g : SupBotHom α β) (a : α) : (f ⊔ g) a = f a ⊔ g a :=
  rfl


@[simp]
theorem bot_apply (a : α) : (⊥ : SupBotHom α β) a = ⊥ :=
  rfl


/-- `Subtype.val` as a `SupBotHom`. -/
def subtypeVal {P : β → Prop}
    (Pbot : P ⊥) (Psup : ∀ ⦃x y : β⦄, P x → P y → P (x ⊔ y)) :
    letI := Subtype.orderBot Pbot
    letI := Subtype.semilatticeSup Psup
    SupBotHom {x : β // P x} β :=
  letI := Subtype.orderBot Pbot
  letI := Subtype.semilatticeSup Psup
                                   /-
                                     F : Type u_1
                                     ι : Type u_2
                                     α : Type u_3
                                     β : Type u_4
                                     γ : Type u_5
                                     δ : Type u_6
                                     inst✝⁴ : FunLike F α β
                                     inst✝³ : Max α
                                     inst✝² : Bot α
                                     inst✝¹ : SemilatticeSup β
                                     inst✝ : OrderBot β
                                     P : β → Prop
                                     Pbot : P Bot.bot
                                     Psup : ∀ ⦃x y : β⦄, P x → P y → P (Max.max x y)
                                     this✝ : OrderBot (Subtype fun x => P x) := Subtype.orderBot Pbot
                                     this : SemilatticeSup (Subtype fun x => P x) := Subtype.semilatticeSup Psup
                                     ⊢ Eq ((SupHom.subtypeVal Psup).toFun Bot.bot) Bot.bot
                                   -/
  .mk (SupHom.subtypeVal Psup) (by simp [Subtype.coe_bot Pbot])
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
lemma subtypeVal_apply {P : β → Prop}
    (Pbot : P ⊥) (Psup : ∀ ⦃x y : β⦄, P x → P y → P (x ⊔ y)) (x : {x : β // P x}) :
    subtypeVal Pbot Psup x = x := rfl


@[simp]
lemma subtypeVal_coe {P : β → Prop}
    (Pbot : P ⊥) (Psup : ∀ ⦃x y : β⦄, P x → P y → P (x ⊔ y)) :
    ⇑(subtypeVal Pbot Psup) = Subtype.val := rfl


/-- Reinterpret an `InfTopHom` as a `TopHom`. -/
def toTopHom (f : InfTopHom α β) : TopHom α β :=
  { f with }


instance : FunLike (InfTopHom α β) α β where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      F : Type u_1
      ι : Type u_2
      α : Type u_3
      β : Type u_4
      γ : Type u_5
      δ : Type u_6
      inst✝⁸ : FunLike F α β
      inst✝⁷ : Min α
      inst✝⁶ : Top α
      inst✝⁵ : Min β
      inst✝⁴ : Top β
      inst✝³ : Min γ
      inst✝² : Top γ
      inst✝¹ : Min δ
      inst✝ : Top δ
      f g : InfTopHom α β
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    obtain ⟨⟨_, _⟩, _⟩ := f
    /-
      case mk.mk
      F : Type u_1
      ι : Type u_2
      α : Type u_3
      β : Type u_4
      γ : Type u_5
      δ : Type u_6
      inst✝⁸ : FunLike F α β
      inst✝⁷ : Min α
      inst✝⁶ : Top α
      inst✝⁵ : Min β
      inst✝⁴ : Top β
      inst✝³ : Min γ
      inst✝² : Top γ
      inst✝¹ : Min δ
      inst✝ : Top δ
      g : InfTopHom α β
      toFun✝ : α → β
      map_inf'✝ : ∀ (a b : α), Eq (toFun✝ (Min.min a b)) (Min.min (toFun✝ a) (toFun✝ …
      map_top'✝ : Eq ({ toFun := toFun✝, map_inf' := map_inf'✝ }.toFun Top.top) Top. …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝, map_inf' := map_inf'✝, map_top'  …
      ⊢ Eq { toFun := toFun✝, map_inf' := map_inf'✝, map_top' := map_top'✝ } g
    -/
    obtain ⟨⟨_, _⟩, _⟩ := g
    /-
      case mk.mk.mk.mk
      F : Type u_1
      ι : Type u_2
      α : Type u_3
      β : Type u_4
      γ : Type u_5
      δ : Type u_6
      inst✝⁸ : FunLike F α β
      inst✝⁷ : Min α
      inst✝⁶ : Top α
      inst✝⁵ : Min β
      inst✝⁴ : Top β
      inst✝³ : Min γ
      inst✝² : Top γ
      inst✝¹ : Min δ
      inst✝ : Top δ
      toFun✝¹ : α → β
      map_inf'✝¹ : ∀ (a b : α), Eq (toFun✝¹ (Min.min a b)) (Min.min (toFun✝¹ a) (toF …
      map_top'✝¹ : Eq ({ toFun := toFun✝¹, map_inf' := map_inf'✝¹ }.toFun Top.top) T …
      toFun✝ : α → β
      map_inf'✝ : ∀ (a b : α), Eq (toFun✝ (Min.min a b)) (Min.min (toFun✝ a) (toFun✝ …
      map_top'✝ : Eq ({ toFun := toFun✝, map_inf' := map_inf'✝ }.toFun Top.top) Top. …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝¹, map_inf' := map_inf'✝¹, map_top …
      ⊢ Eq { toFun := toFun✝¹, map_inf' := map_inf'✝¹, map_top' := map_top'✝¹ } { to …
    -/
    congr
    /-
      🎉 no goals
    -/


instance : InfTopHomClass (InfTopHom α β) α β where
  map_inf f := f.map_inf'
  map_top f := f.map_top'


theorem toFun_eq_coe (f : InfTopHom α β) : f.toFun = f := rfl


@[simp] lemma coe_toInfHom (f : InfTopHom α β) : ⇑f.toInfHom = f := rfl

@[simp] lemma coe_toTopHom (f : InfTopHom α β) : ⇑f.toTopHom = f := rfl

@[simp] lemma coe_mk (f : InfHom α β) (hf) : ⇑(mk f hf) = f := rfl


@[ext]
theorem ext {f g : InfTopHom α β} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


/-- Copy of an `InfTopHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (f : InfTopHom α β) (f' : α → β) (h : f' = f) : InfTopHom α β :=
  { f.toTopHom.copy f' h with toInfHom := f.toInfHom.copy f' h }


@[simp]
theorem coe_copy (f : InfTopHom α β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : InfTopHom α β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- `id` as an `InfTopHom`. -/
@[simps]
protected def id : InfTopHom α α :=
  ⟨InfHom.id α, rfl⟩


instance : Inhabited (InfTopHom α α) :=
  ⟨InfTopHom.id α⟩


@[simp]
theorem coe_id : ⇑(InfTopHom.id α) = id :=
  rfl


@[simp]
theorem id_apply (a : α) : InfTopHom.id α a = a :=
  rfl


/-- Composition of `InfTopHom`s as an `InfTopHom`. -/
def comp (f : InfTopHom β γ) (g : InfTopHom α β) : InfTopHom α γ :=
  { f.toInfHom.comp g.toInfHom, f.toTopHom.comp g.toTopHom with }


@[simp]
theorem coe_comp (f : InfTopHom β γ) (g : InfTopHom α β) : (f.comp g : α → γ) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : InfTopHom β γ) (g : InfTopHom α β) (a : α) : (f.comp g) a = f (g a) :=
  rfl


@[simp]
theorem comp_assoc (f : InfTopHom γ δ) (g : InfTopHom β γ) (h : InfTopHom α β) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp] theorem comp_id (f : InfTopHom α β) : f.comp (InfTopHom.id α) = f := rfl


@[simp] theorem id_comp (f : InfTopHom α β) : (InfTopHom.id β).comp f = f := rfl


@[simp]
theorem cancel_right {g₁ g₂ : InfTopHom β γ} {f : InfTopHom α β} (hf : Surjective f) :
    g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => ext <| hf.forall.2 <| DFunLike.ext_iff.1 h, fun h => congr_arg₂ _ h rfl⟩


@[simp]
theorem cancel_left {g : InfTopHom β γ} {f₁ f₂ : InfTopHom α β} (hg : Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                            /-
                                              α : Type u_3
                                              β : Type u_4
                                              γ : Type u_5
                                              inst✝⁵ : Min α
                                              inst✝⁴ : Top α
                                              inst✝³ : Min β
                                              inst✝² : Top β
                                              inst✝¹ : Min γ
                                              inst✝ : Top γ
                                              g : InfTopHom β γ
                                              f₁ f₂ : InfTopHom α β
                                              hg : Function.Injective ⇑g
                                              h : Eq (g.comp f₁) (g.comp f₂)
                                              a : α
                                              ⊢ Eq (g (f₁ a)) (g (f₂ a))
                                            -/
  ⟨fun h => InfTopHom.ext fun a => hg <| by rw [← comp_apply, h, comp_apply], congr_arg _⟩
                                            /-
                                              🎉 no goals
                                            -/


instance : Min (InfTopHom α β) :=
  ⟨fun f g => { f.toTopHom ⊓ g.toTopHom with toInfHom := f.toInfHom ⊓ g.toInfHom }⟩


instance : SemilatticeInf (InfTopHom α β) :=
  (DFunLike.coe_injective.semilatticeInf _) fun _ _ => rfl


instance : OrderTop (InfTopHom α β) where
  top := ⟨⊤, rfl⟩
  le_top _ _ := le_top


@[simp]
theorem coe_inf (f g : InfTopHom α β) : DFunLike.coe (f ⊓ g) = f ⊓ g :=
  rfl


@[simp]
theorem coe_top : ⇑(⊤ : InfTopHom α β) = ⊤ :=
  rfl


@[simp]
theorem inf_apply (f g : InfTopHom α β) (a : α) : (f ⊓ g) a = f a ⊓ g a :=
  rfl


@[simp]
theorem top_apply (a : α) : (⊤ : InfTopHom α β) a = ⊤ :=
  rfl


/-- `Subtype.val` as an `InfTopHom`. -/
def subtypeVal {P : β → Prop}
    (Ptop : P ⊤) (Pinf : ∀ ⦃x y : β⦄, P x → P y → P (x ⊓ y)) :
    letI := Subtype.orderTop Ptop
    letI := Subtype.semilatticeInf Pinf
    InfTopHom {x : β // P x} β :=
  letI := Subtype.orderTop Ptop
  letI := Subtype.semilatticeInf Pinf
                                   /-
                                     F : Type u_1
                                     ι : Type u_2
                                     α : Type u_3
                                     β : Type u_4
                                     γ : Type u_5
                                     δ : Type u_6
                                     inst✝⁴ : FunLike F α β
                                     inst✝³ : Min α
                                     inst✝² : Top α
                                     inst✝¹ : SemilatticeInf β
                                     inst✝ : OrderTop β
                                     P : β → Prop
                                     Ptop : P Top.top
                                     Pinf : ∀ ⦃x y : β⦄, P x → P y → P (Min.min x y)
                                     this✝ : OrderTop (Subtype fun x => P x) := Subtype.orderTop Ptop
                                     this : SemilatticeInf (Subtype fun x => P x) := Subtype.semilatticeInf Pinf
                                     ⊢ Eq ((InfHom.subtypeVal Pinf).toFun Top.top) Top.top
                                   -/
  .mk (InfHom.subtypeVal Pinf) (by simp [Subtype.coe_top Ptop])
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
lemma subtypeVal_apply {P : β → Prop}
    (Ptop : P ⊤) (Pinf : ∀ ⦃x y : β⦄, P x → P y → P (x ⊓ y)) (x : {x : β // P x}) :
    subtypeVal Ptop Pinf x = x := rfl


@[simp]
lemma subtypeVal_coe {P : β → Prop}
    (Ptop : P ⊤) (Pinf : ∀ ⦃x y : β⦄, P x → P y → P (x ⊓ y)) :
    ⇑(subtypeVal Ptop Pinf) = Subtype.val := rfl


/-- Reinterpret a `LatticeHom` as an `InfHom`. -/
def toInfHom (f : LatticeHom α β) : InfHom α β :=
  { f with }


instance : FunLike (LatticeHom α β) α β where
  coe f := f.toFun
                             /-
                               F : Type u_1
                               ι : Type u_2
                               α : Type u_3
                               β : Type u_4
                               γ : Type u_5
                               δ : Type u_6
                               inst✝⁴ : FunLike F α β
                               inst✝³ : Lattice α
                               inst✝² : Lattice β
                               inst✝¹ : Lattice γ
                               inst✝ : Lattice δ
                               f g : LatticeHom α β
                               h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by obtain ⟨⟨_, _⟩, _⟩ := f; obtain ⟨⟨_, _⟩, _⟩ := g; congr
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


instance : LatticeHomClass (LatticeHom α β) α β where
  map_sup f := f.map_sup'
  map_inf f := f.map_inf'


lemma toFun_eq_coe (f : LatticeHom α β) : f.toFun = f := rfl


@[simp] lemma coe_toSupHom (f : LatticeHom α β) : ⇑f.toSupHom = f := rfl

@[simp] lemma coe_toInfHom (f : LatticeHom α β) : ⇑f.toInfHom = f := rfl

@[ext]
theorem ext {f g : LatticeHom α β} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


/-- Copy of a `LatticeHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (f : LatticeHom α β) (f' : α → β) (h : f' = f) : LatticeHom α β :=
  { f.toSupHom.copy f' h, f.toInfHom.copy f' h with }


@[simp]
theorem coe_copy (f : LatticeHom α β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : LatticeHom α β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- `id` as a `LatticeHom`. -/
protected def id : LatticeHom α α where
  toFun := id
  map_sup' _ _ := rfl
  map_inf' _ _ := rfl


instance : Inhabited (LatticeHom α α) :=
  ⟨LatticeHom.id α⟩


@[simp]
theorem coe_id : ⇑(LatticeHom.id α) = id :=
  rfl


@[simp]
theorem id_apply (a : α) : LatticeHom.id α a = a :=
  rfl


/-- Composition of `LatticeHom`s as a `LatticeHom`. -/
def comp (f : LatticeHom β γ) (g : LatticeHom α β) : LatticeHom α γ :=
  { f.toSupHom.comp g.toSupHom, f.toInfHom.comp g.toInfHom with }


@[simp]
theorem coe_comp (f : LatticeHom β γ) (g : LatticeHom α β) : (f.comp g : α → γ) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : LatticeHom β γ) (g : LatticeHom α β) (a : α) : (f.comp g) a = f (g a) :=
  rfl


@[simp]
-- Porting note: `simp`-normal form of `coe_comp_sup_hom`
theorem coe_comp_sup_hom' (f : LatticeHom β γ) (g : LatticeHom α β) :
    ⟨f ∘ g, map_sup (f.comp g)⟩ = (f : SupHom β γ).comp g :=
  rfl


theorem coe_comp_sup_hom (f : LatticeHom β γ) (g : LatticeHom α β) :
    (f.comp g : SupHom α γ) = (f : SupHom β γ).comp g :=
  rfl


@[simp]
-- Porting note: `simp`-normal form of `coe_comp_inf_hom`
theorem coe_comp_inf_hom' (f : LatticeHom β γ) (g : LatticeHom α β) :
    ⟨f ∘ g, map_inf (f.comp g)⟩ = (f : InfHom β γ).comp g :=
  rfl


theorem coe_comp_inf_hom (f : LatticeHom β γ) (g : LatticeHom α β) :
    (f.comp g : InfHom α γ) = (f : InfHom β γ).comp g :=
  rfl


@[simp]
theorem comp_assoc (f : LatticeHom γ δ) (g : LatticeHom β γ) (h : LatticeHom α β) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp]
theorem comp_id (f : LatticeHom α β) : f.comp (LatticeHom.id α) = f :=
  LatticeHom.ext fun _ => rfl


@[simp]
theorem id_comp (f : LatticeHom α β) : (LatticeHom.id β).comp f = f :=
  LatticeHom.ext fun _ => rfl


@[simp]
theorem cancel_right {g₁ g₂ : LatticeHom β γ} {f : LatticeHom α β} (hf : Surjective f) :
    g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => LatticeHom.ext <| hf.forall.2 <| DFunLike.ext_iff.1 h, fun h => congr_arg₂ _ h rfl⟩


@[simp]
theorem cancel_left {g : LatticeHom β γ} {f₁ f₂ : LatticeHom α β} (hg : Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                             /-
                                               α : Type u_3
                                               β : Type u_4
                                               γ : Type u_5
                                               inst✝² : Lattice α
                                               inst✝¹ : Lattice β
                                               inst✝ : Lattice γ
                                               g : LatticeHom β γ
                                               f₁ f₂ : LatticeHom α β
                                               hg : Function.Injective ⇑g
                                               h : Eq (g.comp f₁) (g.comp f₂)
                                               a : α
                                               ⊢ Eq (g (f₁ a)) (g (f₂ a))
                                             -/
  ⟨fun h => LatticeHom.ext fun a => hg <| by rw [← LatticeHom.comp_apply, h, LatticeHom.comp_apply],
                                             /-
                                               🎉 no goals
                                             -/
    congr_arg _⟩


/-- `Subtype.val` as a `LatticeHom`. -/
def subtypeVal {P : β → Prop}
    (Psup : ∀ ⦃x y⦄, P x → P y → P (x ⊔ y)) (Pinf : ∀ ⦃x y⦄, P x → P y → P (x ⊓ y)) :
    letI := Subtype.lattice Psup Pinf
    LatticeHom {x : β // P x} β :=
  letI := Subtype.lattice Psup Pinf
                                   /-
                                     F : Type u_1
                                     ι : Type u_2
                                     α : Type u_3
                                     β : Type u_4
                                     γ : Type u_5
                                     δ : Type u_6
                                     inst✝⁴ : FunLike F α β
                                     inst✝³ : Lattice α
                                     inst✝² : Lattice β
                                     inst✝¹ : Lattice γ
                                     inst✝ : Lattice δ
                                     P : β → Prop
                                     Psup : ∀ ⦃x y : β⦄, P x → P y → P (Max.max x y)
                                     Pinf : ∀ ⦃x y : β⦄, P x → P y → P (Min.min x y)
                                     this : Lattice (Subtype fun x => P x) := Subtype.lattice Psup Pinf
                                     ⊢ ∀ (a b : Subtype fun x => P x), Eq ((SupHom.subtypeVal Psup).toFun (Min.min  …
                                   -/
  .mk (SupHom.subtypeVal Psup) (by simp [Subtype.coe_inf Pinf])
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
lemma subtypeVal_apply {P : β → Prop}
    (Psup : ∀ ⦃x y⦄, P x → P y → P (x ⊔ y)) (Pinf : ∀ ⦃x y⦄, P x → P y → P (x ⊓ y))
    (x : {x : β // P x}) :
    subtypeVal Psup Pinf x = x := rfl


@[simp]
lemma subtypeVal_coe {P : β → Prop}
    (Psup : ∀ ⦃x y⦄, P x → P y → P (x ⊔ y)) (Pinf : ∀ ⦃x y⦄, P x → P y → P (x ⊓ y)) :
    ⇑(subtypeVal Psup Pinf) = Subtype.val := rfl


/-- An order homomorphism from a linear order is a lattice homomorphism. -/
-- Porting note: made it an `instance` because we're no longer afraid of loops
instance (priority := 100) toLatticeHomClass : LatticeHomClass F α β :=
  { ‹OrderHomClass F α β› with
    map_sup := fun f a b => by
      /-
        F : Type u_1
        ι : Type u_2
        α : Type u_3
        β : Type u_4
        γ : Type u_5
        δ : Type u_6
        inst✝³ : FunLike F α β
        inst✝² : LinearOrder α
        inst✝¹ : Lattice β
        inst✝ : OrderHomClass F α β
        f : F
        a b : α
        ⊢ Eq (f (Max.max a b)) (Max.max (f a) (f b))
      -/
      obtain h | h := le_total a b
        /-
          case inl
          F : Type u_1
          ι : Type u_2
          α : Type u_3
          β : Type u_4
          γ : Type u_5
          δ : Type u_6
          inst✝³ : FunLike F α β
          inst✝² : LinearOrder α
          inst✝¹ : Lattice β
          inst✝ : OrderHomClass F α β
          f : F
          a b : α
          h : LE.le a b
          ⊢ Eq (f (Max.max a b)) (Max.max (f a) (f b))
        -/
      · rw [sup_eq_right.2 h, sup_eq_right.2 (OrderHomClass.mono f h : f a ≤ f b)]
        /-
          🎉 no goals
        -/
        /-
          case inr
          F : Type u_1
          ι : Type u_2
          α : Type u_3
          β : Type u_4
          γ : Type u_5
          δ : Type u_6
          inst✝³ : FunLike F α β
          inst✝² : LinearOrder α
          inst✝¹ : Lattice β
          inst✝ : OrderHomClass F α β
          f : F
          a b : α
          h : LE.le b a
          ⊢ Eq (f (Max.max a b)) (Max.max (f a) (f b))
        -/
      · rw [sup_eq_left.2 h, sup_eq_left.2 (OrderHomClass.mono f h : f b ≤ f a)]
        /-
          🎉 no goals
        -/
    map_inf := fun f a b => by
      /-
        F : Type u_1
        ι : Type u_2
        α : Type u_3
        β : Type u_4
        γ : Type u_5
        δ : Type u_6
        inst✝³ : FunLike F α β
        inst✝² : LinearOrder α
        inst✝¹ : Lattice β
        inst✝ : OrderHomClass F α β
        f : F
        a b : α
        ⊢ Eq (f (Min.min a b)) (Min.min (f a) (f b))
      -/
      obtain h | h := le_total a b
        /-
          case inl
          F : Type u_1
          ι : Type u_2
          α : Type u_3
          β : Type u_4
          γ : Type u_5
          δ : Type u_6
          inst✝³ : FunLike F α β
          inst✝² : LinearOrder α
          inst✝¹ : Lattice β
          inst✝ : OrderHomClass F α β
          f : F
          a b : α
          h : LE.le a b
          ⊢ Eq (f (Min.min a b)) (Min.min (f a) (f b))
        -/
      · rw [inf_eq_left.2 h, inf_eq_left.2 (OrderHomClass.mono f h : f a ≤ f b)]
        /-
          🎉 no goals
        -/
        /-
          case inr
          F : Type u_1
          ι : Type u_2
          α : Type u_3
          β : Type u_4
          γ : Type u_5
          δ : Type u_6
          inst✝³ : FunLike F α β
          inst✝² : LinearOrder α
          inst✝¹ : Lattice β
          inst✝ : OrderHomClass F α β
          f : F
          a b : α
          h : LE.le b a
          ⊢ Eq (f (Min.min a b)) (Min.min (f a) (f b))
        -/
      · rw [inf_eq_right.2 h, inf_eq_right.2 (OrderHomClass.mono f h : f b ≤ f a)] }
        /-
          🎉 no goals
        -/


/-- Reinterpret an order homomorphism to a linear order as a `LatticeHom`. -/
def toLatticeHom (f : F) : LatticeHom α β := f


@[simp]
theorem coe_to_lattice_hom (f : F) : ⇑(toLatticeHom α β f) = f :=
  rfl


@[simp]
theorem to_lattice_hom_apply (f : F) (a : α) : toLatticeHom α β f a = f a :=
  rfl


/-- Reinterpret a `BoundedLatticeHom` as a `SupBotHom`. -/
def toSupBotHom (f : BoundedLatticeHom α β) : SupBotHom α β :=
  { f with }


/-- Reinterpret a `BoundedLatticeHom` as an `InfTopHom`. -/
def toInfTopHom (f : BoundedLatticeHom α β) : InfTopHom α β :=
  { f with }


/-- Reinterpret a `BoundedLatticeHom` as a `BoundedOrderHom`. -/
def toBoundedOrderHom (f : BoundedLatticeHom α β) : BoundedOrderHom α β :=
  { f, (f.toLatticeHom : α →o β) with }


instance instFunLike : FunLike (BoundedLatticeHom α β) α β where
  coe f := f.toFun
                             /-
                               F : Type u_1
                               ι : Type u_2
                               α : Type u_3
                               β : Type u_4
                               γ : Type u_5
                               δ : Type u_6
                               inst✝⁸ : FunLike F α β
                               inst✝⁷ : Lattice α
                               inst✝⁶ : Lattice β
                               inst✝⁵ : Lattice γ
                               inst✝⁴ : Lattice δ
                               inst✝³ : BoundedOrder α
                               inst✝² : BoundedOrder β
                               inst✝¹ : BoundedOrder γ
                               inst✝ : BoundedOrder δ
                               f g : BoundedLatticeHom α β
                               h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by obtain ⟨⟨⟨_, _⟩, _⟩, _⟩ := f; obtain ⟨⟨⟨_, _⟩, _⟩, _⟩ := g; congr
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


instance instBoundedLatticeHomClass : BoundedLatticeHomClass (BoundedLatticeHom α β) α β where
  map_sup f := f.map_sup'
  map_inf f := f.map_inf'
  map_top f := f.map_top'
  map_bot f := f.map_bot'


@[simp] lemma toFun_eq_coe (f : BoundedLatticeHom α β) : f.toFun = f := rfl


@[simp] lemma coe_toLatticeHom (f : BoundedLatticeHom α β) : ⇑f.toLatticeHom = f := rfl

@[simp] lemma coe_toSupBotHom (f : BoundedLatticeHom α β) : ⇑f.toSupBotHom = f := rfl

@[simp] lemma coe_toInfTopHom (f : BoundedLatticeHom α β) : ⇑f.toInfTopHom = f := rfl

@[simp] lemma coe_toBoundedOrderHom (f : BoundedLatticeHom α β) : ⇑f.toBoundedOrderHom = f := rfl

@[simp] lemma coe_mk (f : LatticeHom α β) (hf hf') : ⇑(mk f hf hf') = f := rfl


@[ext]
theorem ext {f g : BoundedLatticeHom α β} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


/-- Copy of a `BoundedLatticeHom` with a new `toFun` equal to the old one. Useful to fix
definitional equalities. -/
protected def copy (f : BoundedLatticeHom α β) (f' : α → β) (h : f' = f) : BoundedLatticeHom α β :=
  { f.toLatticeHom.copy f' h, f.toBoundedOrderHom.copy f' h with }


@[simp]
theorem coe_copy (f : BoundedLatticeHom α β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : BoundedLatticeHom α β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- `id` as a `BoundedLatticeHom`. -/
protected def id : BoundedLatticeHom α α :=
  { LatticeHom.id α, BoundedOrderHom.id α with }


instance : Inhabited (BoundedLatticeHom α α) :=
  ⟨BoundedLatticeHom.id α⟩


@[simp]
theorem coe_id : ⇑(BoundedLatticeHom.id α) = id :=
  rfl


@[simp]
theorem id_apply (a : α) : BoundedLatticeHom.id α a = a :=
  rfl


/-- Composition of `BoundedLatticeHom`s as a `BoundedLatticeHom`. -/
def comp (f : BoundedLatticeHom β γ) (g : BoundedLatticeHom α β) : BoundedLatticeHom α γ :=
  { f.toLatticeHom.comp g.toLatticeHom, f.toBoundedOrderHom.comp g.toBoundedOrderHom with }


@[simp]
theorem coe_comp (f : BoundedLatticeHom β γ) (g : BoundedLatticeHom α β) :
    (f.comp g : α → γ) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : BoundedLatticeHom β γ) (g : BoundedLatticeHom α β) (a : α) :
    (f.comp g) a = f (g a) :=
  rfl


@[simp]
-- Porting note: `simp`-normal form of `coe_comp_lattice_hom`
theorem coe_comp_lattice_hom' (f : BoundedLatticeHom β γ) (g : BoundedLatticeHom α β) :
    (⟨(f : SupHom β γ).comp g, map_inf (f.comp g)⟩ : LatticeHom α γ) =
      (f : LatticeHom β γ).comp g :=
  rfl


theorem coe_comp_lattice_hom (f : BoundedLatticeHom β γ) (g : BoundedLatticeHom α β) :
    (f.comp g : LatticeHom α γ) = (f : LatticeHom β γ).comp g :=
  rfl


@[simp]
-- Porting note: `simp`-normal form of `coe_comp_sup_hom`
theorem coe_comp_sup_hom' (f : BoundedLatticeHom β γ) (g : BoundedLatticeHom α β) :
    ⟨f ∘ g, map_sup (f.comp g)⟩ = (f : SupHom β γ).comp g :=
  rfl


theorem coe_comp_sup_hom (f : BoundedLatticeHom β γ) (g : BoundedLatticeHom α β) :
    (f.comp g : SupHom α γ) = (f : SupHom β γ).comp g :=
  rfl


@[simp]
-- Porting note: `simp`-normal form of `coe_comp_inf_hom`
theorem coe_comp_inf_hom' (f : BoundedLatticeHom β γ) (g : BoundedLatticeHom α β) :
    ⟨f ∘ g, map_inf (f.comp g)⟩ = (f : InfHom β γ).comp g :=
  rfl


theorem coe_comp_inf_hom (f : BoundedLatticeHom β γ) (g : BoundedLatticeHom α β) :
    (f.comp g : InfHom α γ) = (f : InfHom β γ).comp g :=
  rfl


@[simp]
theorem comp_assoc (f : BoundedLatticeHom γ δ) (g : BoundedLatticeHom β γ)
    (h : BoundedLatticeHom α β) : (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp] theorem comp_id (f : BoundedLatticeHom α β) : f.comp (BoundedLatticeHom.id α) = f := rfl


@[simp] theorem id_comp (f : BoundedLatticeHom α β) : (BoundedLatticeHom.id β).comp f = f := rfl


@[simp]
theorem cancel_right {g₁ g₂ : BoundedLatticeHom β γ} {f : BoundedLatticeHom α β}
    (hf : Surjective f) : g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => BoundedLatticeHom.ext <| hf.forall.2 <| DFunLike.ext_iff.1 h,
    fun h => congr_arg₂ _ h rfl⟩


@[simp]
theorem cancel_left {g : BoundedLatticeHom β γ} {f₁ f₂ : BoundedLatticeHom α β} (hg : Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                  /-
                                    α : Type u_3
                                    β : Type u_4
                                    γ : Type u_5
                                    inst✝⁵ : Lattice α
                                    inst✝⁴ : Lattice β
                                    inst✝³ : Lattice γ
                                    inst✝² : BoundedOrder α
                                    inst✝¹ : BoundedOrder β
                                    inst✝ : BoundedOrder γ
                                    g : BoundedLatticeHom β γ
                                    f₁ f₂ : BoundedLatticeHom α β
                                    hg : Function.Injective ⇑g
                                    h : Eq (g.comp f₁) (g.comp f₂)
                                    a : α
                                    ⊢ Eq (g (f₁ a)) (g (f₂ a))
                                  -/
  ⟨fun h => ext fun a => hg <| by rw [← comp_apply, h, comp_apply], congr_arg _⟩
                                  /-
                                    🎉 no goals
                                  -/


/-- `Subtype.val` as a `BoundedLatticeHom`. -/
def subtypeVal {P : β → Prop} (Pbot : P ⊥) (Ptop : P ⊤)
    (Psup : ∀ ⦃x y⦄, P x → P y → P (x ⊔ y)) (Pinf : ∀ ⦃x y⦄, P x → P y → P (x ⊓ y)) :
    letI := Subtype.lattice Psup Pinf
    letI := Subtype.boundedOrder Pbot Ptop
    BoundedLatticeHom {x : β // P x} β :=
  letI := Subtype.lattice Psup Pinf
  letI := Subtype.boundedOrder Pbot Ptop
                                  /-
                                    F : Type u_1
                                    ι : Type u_2
                                    α : Type u_3
                                    β : Type u_4
                                    γ : Type u_5
                                    δ : Type u_6
                                    inst✝⁸ : FunLike F α β
                                    inst✝⁷ : Lattice α
                                    inst✝⁶ : Lattice β
                                    inst✝⁵ : Lattice γ
                                    inst✝⁴ : Lattice δ
                                    inst✝³ : BoundedOrder α
                                    inst✝² : BoundedOrder β
                                    inst✝¹ : BoundedOrder γ
                                    inst✝ : BoundedOrder δ
                                    P : β → Prop
                                    Pbot : P Bot.bot
                                    Ptop : P Top.top
                                    Psup : ∀ ⦃x y : β⦄, P x → P y → P (Max.max x y)
                                    Pinf : ∀ ⦃x y : β⦄, P x → P y → P (Min.min x y)
                                    this✝ : Lattice (Subtype fun x => P x) := Subtype.lattice Psup Pinf
                                    this : BoundedOrder (Subtype P) := Subtype.boundedOrder Pbot Ptop
                                    ⊢ Eq ((LatticeHom.subtypeVal Psup Pinf).toFun Top.top) Top.top
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
  .mk (.subtypeVal Psup Pinf) (by simp [Subtype.coe_top Ptop]) (by simp [Subtype.coe_bot Pbot])
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
lemma subtypeVal_apply {P : β → Prop}
    (Pbot : P ⊥) (Ptop : P ⊤) (Psup : ∀ ⦃x y⦄, P x → P y → P (x ⊔ y))
    (Pinf : ∀ ⦃x y⦄, P x → P y → P (x ⊓ y)) (x : {x : β // P x}) :
    subtypeVal Pbot Ptop Psup Pinf x = x := rfl


@[simp]
lemma subtypeVal_coe {P : β → Prop} (Pbot : P ⊥) (Ptop : P ⊤)
    (Psup : ∀ ⦃x y⦄, P x → P y → P (x ⊔ y)) (Pinf : ∀ ⦃x y⦄, P x → P y → P (x ⊓ y)) :
    ⇑(subtypeVal Pbot Ptop Psup Pinf) = Subtype.val := rfl


/-- Reinterpret a supremum homomorphism as an infimum homomorphism between the dual lattices. -/
@[simps]
protected def dual : SupHom α β ≃ InfHom αᵒᵈ βᵒᵈ where
  toFun f := ⟨f, f.map_sup'⟩
  invFun f := ⟨f, f.map_inf'⟩
  left_inv _ := rfl
  right_inv _ := rfl


@[simp]
theorem dual_id : SupHom.dual (SupHom.id α) = InfHom.id _ :=
  rfl


@[simp]
theorem dual_comp (g : SupHom β γ) (f : SupHom α β) :
    SupHom.dual (g.comp f) = (SupHom.dual g).comp (SupHom.dual f) :=
  rfl


@[simp]
theorem symm_dual_id : SupHom.dual.symm (InfHom.id _) = SupHom.id α :=
  rfl


@[simp]
theorem symm_dual_comp (g : InfHom βᵒᵈ γᵒᵈ) (f : InfHom αᵒᵈ βᵒᵈ) :
    SupHom.dual.symm (g.comp f) =
      (SupHom.dual.symm g).comp (SupHom.dual.symm f) :=
  rfl


/-- Reinterpret an infimum homomorphism as a supremum homomorphism between the dual lattices. -/
@[simps]
protected def dual : InfHom α β ≃ SupHom αᵒᵈ βᵒᵈ where
  toFun f := ⟨f, f.map_inf'⟩
  invFun f := ⟨f, f.map_sup'⟩
  left_inv _ := rfl
  right_inv _ := rfl


@[simp]
theorem dual_id : InfHom.dual (InfHom.id α) = SupHom.id _ :=
  rfl


@[simp]
theorem dual_comp (g : InfHom β γ) (f : InfHom α β) :
    InfHom.dual (g.comp f) = (InfHom.dual g).comp (InfHom.dual f) :=
  rfl


@[simp]
theorem symm_dual_id : InfHom.dual.symm (SupHom.id _) = InfHom.id α :=
  rfl


@[simp]
theorem symm_dual_comp (g : SupHom βᵒᵈ γᵒᵈ) (f : SupHom αᵒᵈ βᵒᵈ) :
    InfHom.dual.symm (g.comp f) =
      (InfHom.dual.symm g).comp (InfHom.dual.symm f) :=
  rfl


/-- Reinterpret a finitary supremum homomorphism as a finitary infimum homomorphism between the dual
lattices. -/
def dual : SupBotHom α β ≃ InfTopHom αᵒᵈ βᵒᵈ where
  toFun f := ⟨SupHom.dual f.toSupHom, f.map_bot'⟩
  invFun f := ⟨SupHom.dual.symm f.toInfHom, f.map_top'⟩
  left_inv _ := rfl
  right_inv _ := rfl


@[simp] theorem dual_id : SupBotHom.dual (SupBotHom.id α) = InfTopHom.id _ := rfl


@[simp]
theorem dual_comp (g : SupBotHom β γ) (f : SupBotHom α β) :
    SupBotHom.dual (g.comp f) = (SupBotHom.dual g).comp (SupBotHom.dual f) :=
  rfl


@[simp]
theorem symm_dual_id : SupBotHom.dual.symm (InfTopHom.id _) = SupBotHom.id α :=
  rfl


@[simp]
theorem symm_dual_comp (g : InfTopHom βᵒᵈ γᵒᵈ) (f : InfTopHom αᵒᵈ βᵒᵈ) :
    SupBotHom.dual.symm (g.comp f) =
      (SupBotHom.dual.symm g).comp (SupBotHom.dual.symm f) :=
  rfl


/-- Reinterpret a finitary infimum homomorphism as a finitary supremum homomorphism between the dual
lattices. -/
@[simps]
protected def dual : InfTopHom α β ≃ SupBotHom αᵒᵈ βᵒᵈ where
  toFun f := ⟨InfHom.dual f.toInfHom, f.map_top'⟩
  invFun f := ⟨InfHom.dual.symm f.toSupHom, f.map_bot'⟩
  left_inv _ := rfl
  right_inv _ := rfl


@[simp]
theorem dual_id : InfTopHom.dual (InfTopHom.id α) = SupBotHom.id _ :=
  rfl


@[simp]
theorem dual_comp (g : InfTopHom β γ) (f : InfTopHom α β) :
    InfTopHom.dual (g.comp f) = (InfTopHom.dual g).comp (InfTopHom.dual f) :=
  rfl


@[simp]
theorem symm_dual_id : InfTopHom.dual.symm (SupBotHom.id _) = InfTopHom.id α :=
  rfl


@[simp]
theorem symm_dual_comp (g : SupBotHom βᵒᵈ γᵒᵈ) (f : SupBotHom αᵒᵈ βᵒᵈ) :
    InfTopHom.dual.symm (g.comp f) =
      (InfTopHom.dual.symm g).comp (InfTopHom.dual.symm f) :=
  rfl


/-- Reinterpret a lattice homomorphism as a lattice homomorphism between the dual lattices. -/
@[simps]
protected def dual : LatticeHom α β ≃ LatticeHom αᵒᵈ βᵒᵈ where
  toFun f := ⟨InfHom.dual f.toInfHom, f.map_sup'⟩
  invFun f := ⟨SupHom.dual.symm f.toInfHom, f.map_sup'⟩
  left_inv _ := rfl
  right_inv _ := rfl


@[simp] theorem dual_id : LatticeHom.dual (LatticeHom.id α) = LatticeHom.id _ := rfl


@[simp]
theorem dual_comp (g : LatticeHom β γ) (f : LatticeHom α β) :
    LatticeHom.dual (g.comp f) = (LatticeHom.dual g).comp (LatticeHom.dual f) :=
  rfl


@[simp]
theorem symm_dual_id : LatticeHom.dual.symm (LatticeHom.id _) = LatticeHom.id α :=
  rfl


@[simp]
theorem symm_dual_comp (g : LatticeHom βᵒᵈ γᵒᵈ) (f : LatticeHom αᵒᵈ βᵒᵈ) :
    LatticeHom.dual.symm (g.comp f) =
      (LatticeHom.dual.symm g).comp (LatticeHom.dual.symm f) :=
  rfl


/-- Reinterpret a bounded lattice homomorphism as a bounded lattice homomorphism between the dual
bounded lattices. -/
@[simps]
protected def dual : BoundedLatticeHom α β ≃ BoundedLatticeHom αᵒᵈ βᵒᵈ where
  toFun f := ⟨LatticeHom.dual f.toLatticeHom, f.map_bot', f.map_top'⟩
  invFun f := ⟨LatticeHom.dual.symm f.toLatticeHom, f.map_bot', f.map_top'⟩
  left_inv _ := rfl
  right_inv _ := rfl


@[simp]
theorem dual_id : BoundedLatticeHom.dual (BoundedLatticeHom.id α) = BoundedLatticeHom.id _ :=
  rfl


@[simp]
theorem dual_comp (g : BoundedLatticeHom β γ) (f : BoundedLatticeHom α β) :
    BoundedLatticeHom.dual (g.comp f) =
      (BoundedLatticeHom.dual g).comp (BoundedLatticeHom.dual f) :=
  rfl


@[simp]
theorem symm_dual_id :
    BoundedLatticeHom.dual.symm (BoundedLatticeHom.id _) = BoundedLatticeHom.id α :=
  rfl


@[simp]
theorem symm_dual_comp (g : BoundedLatticeHom βᵒᵈ γᵒᵈ) (f : BoundedLatticeHom αᵒᵈ βᵒᵈ) :
    BoundedLatticeHom.dual.symm (g.comp f) =
      (BoundedLatticeHom.dual.symm g).comp (BoundedLatticeHom.dual.symm f) :=
  rfl


/-- Natural projection homomorphism from `α × β` to `α`. -/
def fst : LatticeHom (α × β) α where
  toFun := Prod.fst
  map_sup' _ _ := rfl
  map_inf' _ _ := rfl


/-- Natural projection homomorphism from `α × β` to `β`. -/
def snd : LatticeHom (α × β) β where
  toFun := Prod.snd
  map_sup' _ _ := rfl
  map_inf' _ _ := rfl


@[simp, norm_cast] lemma coe_fst : ⇑(fst (α := α) (β := β)) = Prod.fst := rfl

@[simp, norm_cast] lemma coe_snd : ⇑(snd (α := α) (β := β)) = Prod.snd := rfl

lemma fst_apply (x : α × β) : fst x = x.fst := rfl

lemma snd_apply (x : α × β) : snd x = x.snd := rfl


/-- Evaluation as a lattice homomorphism. -/
def evalLatticeHom (i : ι) : LatticeHom (∀ i, α i) (α i) where
  toFun := Function.eval i
  map_sup' _a _b := rfl
  map_inf' _a _b := rfl


@[simp, norm_cast]
lemma coe_evalLatticeHom (i : ι) : ⇑(evalLatticeHom (α := α) i) = Function.eval i := rfl


lemma evalLatticeHom_apply (i : ι) (f : ∀ i, α i) : evalLatticeHom i f = f i := rfl


/-- Adjoins a `⊤` to the domain and codomain of a `SupHom`. -/
@[simps]
protected def withTop (f : SupHom α β) : SupHom (WithTop α) (WithTop β) where
  -- Porting note: this was `Option.map f`
  toFun := WithTop.map f
  map_sup' a b :=
    match a, b with
    | ⊤, ⊤ => rfl
    | ⊤, (b : α) => rfl
    | (a : α), ⊤ => rfl
    | (a : α), (b : α) => congr_arg _ (f.map_sup' _ _)


@[simp]
theorem withTop_id : (SupHom.id α).withTop = SupHom.id _ := DFunLike.coe_injective Option.map_id


@[simp]
theorem withTop_comp (f : SupHom β γ) (g : SupHom α β) :
    (f.comp g).withTop = f.withTop.comp g.withTop :=
  DFunLike.coe_injective <| Eq.symm <| Option.map_comp_map _ _


/-- Adjoins a `⊥` to the domain and codomain of a `SupHom`. -/
@[simps]
protected def withBot (f : SupHom α β) : SupBotHom (WithBot α) (WithBot β) where
  toFun := Option.map f
  map_sup' a b :=
    match a, b with
    | ⊥, ⊥ => rfl
    | ⊥, (b : α) => rfl
    | (a : α), ⊥ => rfl
    | (a : α), (b : α) => congr_arg _ (f.map_sup' _ _)
  map_bot' := rfl


@[simp]
theorem withBot_id : (SupHom.id α).withBot = SupBotHom.id _ := DFunLike.coe_injective Option.map_id


@[simp]
theorem withBot_comp (f : SupHom β γ) (g : SupHom α β) :
    (f.comp g).withBot = f.withBot.comp g.withBot :=
  DFunLike.coe_injective <| Eq.symm <| Option.map_comp_map _ _


/-- Adjoins a `⊤` to the codomain of a `SupHom`. -/
@[simps]
def withTop' [OrderTop β] (f : SupHom α β) : SupHom (WithTop α) β where
  toFun a := a.elim ⊤ f
  map_sup' a b :=
    match a, b with
    | ⊤, ⊤ => (top_sup_eq _).symm
    | ⊤, (b : α) => (top_sup_eq _).symm
    | (a : α), ⊤ => (sup_top_eq _).symm
    | (a : α), (b : α) => f.map_sup' _ _


/-- Adjoins a `⊥` to the domain of a `SupHom`. -/
@[simps]
def withBot' [OrderBot β] (f : SupHom α β) : SupBotHom (WithBot α) β where
  toFun a := a.elim ⊥ f
  map_sup' a b :=
    match a, b with
    | ⊥, ⊥ => (bot_sup_eq _).symm
    | ⊥, (b : α) => (bot_sup_eq _).symm
    | (a : α), ⊥ => (sup_bot_eq _).symm
    | (a : α), (b : α) => f.map_sup' _ _
  map_bot' := rfl


/-- Adjoins a `⊤` to the domain and codomain of an `InfHom`. -/
@[simps]
protected def withTop (f : InfHom α β) : InfTopHom (WithTop α) (WithTop β) where
  toFun := Option.map f
  map_inf' a b :=
    match a, b with
    | ⊤, ⊤ => rfl
    | ⊤, (b : α) => rfl
    | (a : α), ⊤ => rfl
    | (a : α), (b : α) => congr_arg _ (f.map_inf' _ _)
  map_top' := rfl


@[simp]
theorem withTop_id : (InfHom.id α).withTop = InfTopHom.id _ := DFunLike.coe_injective Option.map_id


@[simp]
theorem withTop_comp (f : InfHom β γ) (g : InfHom α β) :
    (f.comp g).withTop = f.withTop.comp g.withTop :=
  DFunLike.coe_injective <| Eq.symm <| Option.map_comp_map _ _


/-- Adjoins a `⊥` to the domain and codomain of an `InfHom`. -/
@[simps]
protected def withBot (f : InfHom α β) : InfHom (WithBot α) (WithBot β) where
  toFun := Option.map f
  map_inf' a b :=
    match a, b with
    | ⊥, ⊥ => rfl
    | ⊥, (b : α) => rfl
    | (a : α), ⊥ => rfl
    | (a : α), (b : α) => congr_arg _ (f.map_inf' _ _)


@[simp]
theorem withBot_id : (InfHom.id α).withBot = InfHom.id _ := DFunLike.coe_injective Option.map_id


@[simp]
theorem withBot_comp (f : InfHom β γ) (g : InfHom α β) :
    (f.comp g).withBot = f.withBot.comp g.withBot :=
  DFunLike.coe_injective <| Eq.symm <| Option.map_comp_map _ _


/-- Adjoins a `⊤` to the codomain of an `InfHom`. -/
@[simps]
def withTop' [OrderTop β] (f : InfHom α β) : InfTopHom (WithTop α) β where
  toFun a := a.elim ⊤ f
  map_inf' a b :=
    match a, b with
    | ⊤, ⊤ => (top_inf_eq _).symm
    | ⊤, (b : α) => (top_inf_eq _).symm
    | (a : α), ⊤ => (inf_top_eq _).symm
    | (a : α), (b : α) => f.map_inf' _ _
  map_top' := rfl


/-- Adjoins a `⊥` to the codomain of an `InfHom`. -/
@[simps]
def withBot' [OrderBot β] (f : InfHom α β) : InfHom (WithBot α) β where
  toFun a := a.elim ⊥ f
  map_inf' a b :=
    match a, b with
    | ⊥, ⊥ => (bot_inf_eq _).symm
    | ⊥, (b : α) => (bot_inf_eq _).symm
    | (a : α), ⊥ => (inf_bot_eq _).symm
    | (a : α), (b : α) => f.map_inf' _ _


/-- Adjoins a `⊤` to the domain and codomain of a `LatticeHom`. -/
@[simps]
protected def withTop (f : LatticeHom α β) : LatticeHom (WithTop α) (WithTop β) :=
  { f.toInfHom.withTop with toSupHom := f.toSupHom.withTop }

-- Porting note: `simps` doesn't generate those

@[simp, norm_cast]
lemma coe_withTop (f : LatticeHom α β) : ⇑f.withTop = WithTop.map f := rfl


lemma withTop_apply (f : LatticeHom α β) (a : WithTop α) : f.withTop a = a.map f := rfl


@[simp]
theorem withTop_id : (LatticeHom.id α).withTop = LatticeHom.id _ :=
  DFunLike.coe_injective Option.map_id


@[simp]
theorem withTop_comp (f : LatticeHom β γ) (g : LatticeHom α β) :
    (f.comp g).withTop = f.withTop.comp g.withTop :=
  DFunLike.coe_injective <| Eq.symm <| Option.map_comp_map _ _


/-- Adjoins a `⊥` to the domain and codomain of a `LatticeHom`. -/
@[simps]
protected def withBot (f : LatticeHom α β) : LatticeHom (WithBot α) (WithBot β) :=
  { f.toInfHom.withBot with toSupHom := f.toSupHom.withBot }

-- Porting note: `simps` doesn't generate those

@[simp, norm_cast]
lemma coe_withBot (f : LatticeHom α β) : ⇑f.withBot = Option.map f := rfl


lemma withBot_apply (f : LatticeHom α β) (a : WithBot α) : f.withBot a = a.map f := rfl


@[simp]
theorem withBot_id : (LatticeHom.id α).withBot = LatticeHom.id _ :=
  DFunLike.coe_injective Option.map_id


@[simp]
theorem withBot_comp (f : LatticeHom β γ) (g : LatticeHom α β) :
    (f.comp g).withBot = f.withBot.comp g.withBot :=
  DFunLike.coe_injective <| Eq.symm <| Option.map_comp_map _ _


/-- Adjoins a `⊤` and `⊥` to the domain and codomain of a `LatticeHom`. -/
@[simps]
def withTopWithBot (f : LatticeHom α β) :
    BoundedLatticeHom (WithTop <| WithBot α) (WithTop <| WithBot β) :=
  ⟨f.withBot.withTop, rfl, rfl⟩

-- Porting note: `simps` doesn't generate those

@[simp, norm_cast]
lemma coe_withTopWithBot (f : LatticeHom α β) : ⇑f.withTopWithBot = Option.map (Option.map f) := rfl


lemma withTopWithBot_apply (f : LatticeHom α β) (a : WithTop <| WithBot α) :
    f.withTopWithBot a = a.map (Option.map f) := rfl


@[simp]
theorem withTopWithBot_id : (LatticeHom.id α).withTopWithBot = BoundedLatticeHom.id _ :=
  DFunLike.coe_injective <| by
    /-
      α : Type u_3
      inst✝ : Lattice α
      ⊢ Eq ((fun f => ⇑f) (LatticeHom.id α).withTopWithBot) ((fun f => ⇑f) (BoundedL …
    -/
    refine (congr_arg Option.map ?_).trans Option.map_id
    /-
      α : Type u_3
      inst✝ : Lattice α
      ⊢ Eq (⇑(LatticeHom.id α).withBot.toSupHom) id
    -/
    rw [withBot_id]
    /-
      α : Type u_3
      inst✝ : Lattice α
      ⊢ Eq (⇑(LatticeHom.id (WithBot α)).toSupHom) id
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem withTopWithBot_comp (f : LatticeHom β γ) (g : LatticeHom α β) :
    (f.comp g).withTopWithBot = f.withTopWithBot.comp g.withTopWithBot := by
  /-
    α : Type u_3
    β : Type u_4
    γ : Type u_5
    inst✝² : Lattice α
    inst✝¹ : Lattice β
    inst✝ : Lattice γ
    f : LatticeHom β γ
    g : LatticeHom α β
    ⊢ Eq (f.comp g).withTopWithBot (f.withTopWithBot.comp g.withTopWithBot)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- Adjoins a `⊥` to the codomain of a `LatticeHom`. -/
@[simps]
def withTop' [OrderTop β] (f : LatticeHom α β) : LatticeHom (WithTop α) β :=
  { f.toSupHom.withTop', f.toInfHom.withTop' with }


/-- Adjoins a `⊥` to the domain and codomain of a `LatticeHom`. -/
@[simps]
def withBot' [OrderBot β] (f : LatticeHom α β) : LatticeHom (WithBot α) β :=
  { f.toSupHom.withBot', f.toInfHom.withBot' with }


/-- Adjoins a `⊤` and `⊥` to the codomain of a `LatticeHom`. -/
@[simps]
def withTopWithBot' [BoundedOrder β] (f : LatticeHom α β) :
    BoundedLatticeHom (WithTop <| WithBot α) β where
  toLatticeHom := f.withBot'.withTop'
  map_top' := rfl
  map_bot' := rfl


