/-- The category of types with an omega complete partial order. -/
def ωCPO : Type (u + 1) :=
  Bundled OmegaCompletePartialOrder


instance : BundledHom @ContinuousHom where
  toFun := @ContinuousHom.Simps.apply
  id := @ContinuousHom.id
  comp := @ContinuousHom.comp
  hom_ext := @ContinuousHom.coe_inj

-- Porting note: `deriving instance ConcreteCategory` didn't work.

deriving instance LargeCategory for ωCPO

                                       /-
                                         ⊢ CategoryTheory.ConcreteCategory ωCPO
                                       -/
instance : ConcreteCategory ωCPO := by unfold ωCPO; infer_instance
                                                    /-
                                                      🎉 no goals
                                                    -/


instance : CoeSort ωCPO Type* :=
  Bundled.coeSort


/-- Construct a bundled ωCPO from the underlying type and typeclass. -/
def of (α : Type*) [OmegaCompletePartialOrder α] : ωCPO :=
  Bundled.of α


@[simp]
theorem coe_of (α : Type*) [OmegaCompletePartialOrder α] : ↥(of α) = α :=
  rfl


instance : Inhabited ωCPO :=
  ⟨of PUnit⟩


instance (α : ωCPO) : OmegaCompletePartialOrder α :=
  α.str


/-- The pi-type gives a cone for a product. -/
def product {J : Type v} (f : J → ωCPO.{v}) : Fan f :=
  Fan.mk (of (∀ j, f j)) fun j => .mk (Pi.evalOrderHom j) fun _ => rfl


/-- The pi-type is a limit cone for the product. -/
def isProduct (J : Type v) (f : J → ωCPO) : IsLimit (product f) where
  lift s :=
    -- Porting note: Original proof didn't have `.toFun`
    ⟨⟨fun t j => (s.π.app ⟨j⟩).toFun t, fun _ _ h j => (s.π.app ⟨j⟩).monotone h⟩,
      fun x => funext fun j => (s.π.app ⟨j⟩).continuous x⟩
  uniq s m w := by
    /-
      J : Type v
      f : J → ωCPO
      s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
      m : Quiver.Hom s.pt (ωCPO.HasProducts.product f).pt
      w : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
      ⊢ Eq m ((fun s => { toFun := fun t j => (s.π.app { as := j }).toFun t, monoton …
    -/
    ext t; funext j -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext t j`
    /-
      case w.h
      J : Type v
      f : J → ωCPO
      s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
      m : Quiver.Hom s.pt (ωCPO.HasProducts.product f).pt
      w : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
      t : (CategoryTheory.forget ωCPO).obj s.pt
      j : J
      ⊢ Eq (m t j) (((fun s => { toFun := fun t j => (s.π.app { as := j }).toFun t,  …
    -/
    change m.toFun t j = (s.π.app ⟨j⟩).toFun t
    /-
      case w.h
      J : Type v
      f : J → ωCPO
      s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
      m : Quiver.Hom s.pt (ωCPO.HasProducts.product f).pt
      w : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
      t : (CategoryTheory.forget ωCPO).obj s.pt
      j : J
      ⊢ Eq (m.toFun t j) ((s.π.app { as := j }).toFun t)
    -/
    rw [← w ⟨j⟩]
    /-
      case w.h
      J : Type v
      f : J → ωCPO
      s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
      m : Quiver.Hom s.pt (ωCPO.HasProducts.product f).pt
      w : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
      t : (CategoryTheory.forget ωCPO).obj s.pt
      j : J
      ⊢ Eq (m.toFun t j) ((CategoryTheory.CategoryStruct.comp m ((ωCPO.HasProducts.p …
    -/
    rfl
    /-
      🎉 no goals
    -/
  fac _ _ := rfl


instance (J : Type v) (f : J → ωCPO.{v}) : HasProduct f :=
  HasLimit.mk ⟨_, isProduct _ f⟩


instance omegaCompletePartialOrderEqualizer {α β : Type*} [OmegaCompletePartialOrder α]
    [OmegaCompletePartialOrder β] (f g : α →𝒄 β) :
    OmegaCompletePartialOrder { a : α // f a = g a } :=
  OmegaCompletePartialOrder.subtype _ fun c hc => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : OmegaCompletePartialOrder α
      inst✝ : OmegaCompletePartialOrder β
      f g : OmegaCompletePartialOrder.ContinuousHom α β
      c : OmegaCompletePartialOrder.Chain α
      hc : ∀ (i : α), Membership.mem c i → Eq (f i) (g i)
      ⊢ Eq (f (OmegaCompletePartialOrder.ωSup c)) (g (OmegaCompletePartialOrder.ωSup …
    -/
    rw [f.continuous, g.continuous]
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : OmegaCompletePartialOrder α
      inst✝ : OmegaCompletePartialOrder β
      f g : OmegaCompletePartialOrder.ContinuousHom α β
      c : OmegaCompletePartialOrder.Chain α
      hc : ∀ (i : α), Membership.mem c i → Eq (f i) (g i)
      ⊢ Eq (OmegaCompletePartialOrder.ωSup (c.map ↑f)) (OmegaCompletePartialOrder.ωS …
    -/
    congr 1
    /-
      case e_a
      α : Type u_1
      β : Type u_2
      inst✝¹ : OmegaCompletePartialOrder α
      inst✝ : OmegaCompletePartialOrder β
      f g : OmegaCompletePartialOrder.ContinuousHom α β
      c : OmegaCompletePartialOrder.Chain α
      hc : ∀ (i : α), Membership.mem c i → Eq (f i) (g i)
      ⊢ Eq (c.map ↑f) (c.map ↑g)
    -/
    apply OrderHom.ext; funext x -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext`
    /-
      case e_a.h.h
      α : Type u_1
      β : Type u_2
      inst✝¹ : OmegaCompletePartialOrder α
      inst✝ : OmegaCompletePartialOrder β
      f g : OmegaCompletePartialOrder.ContinuousHom α β
      c : OmegaCompletePartialOrder.Chain α
      hc : ∀ (i : α), Membership.mem c i → Eq (f i) (g i)
      x : Nat
      ⊢ Eq ((c.map ↑f) x) ((c.map ↑g) x)
    -/
    apply hc _ ⟨_, rfl⟩
    /-
      🎉 no goals
    -/


/-- The equalizer inclusion function as a `ContinuousHom`. -/
def equalizerι {α β : Type*} [OmegaCompletePartialOrder α] [OmegaCompletePartialOrder β]
    (f g : α →𝒄 β) : { a : α // f a = g a } →𝒄 α :=
  .mk (OrderHom.Subtype.val _) fun _ => rfl


/-- A construction of the equalizer fork. -/
-- Porting note: Changed `{ a // f a = g a }` to `{ a // f.toFun a = g.toFun a }`
def equalizer {X Y : ωCPO.{v}} (f g : X ⟶ Y) : Fork f g :=
  Fork.ofι (P := ωCPO.of { a // f.toFun a = g.toFun a }) (equalizerι f g)
    (ContinuousHom.ext _ _ fun x => x.2)


/-- The equalizer fork is a limit. -/
def isEqualizer {X Y : ωCPO.{v}} (f g : X ⟶ Y) : IsLimit (equalizer f g) :=
  Fork.IsLimit.mk' _ fun s =>
    -- Porting note: Changed `s.ι x` to `s.ι.toFun x`
                                           /-
                                             X Y : ωCPO
                                             f g : Quiver.Hom X Y
                                             s : CategoryTheory.Limits.Fork f g
                                             x : ↑(((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingParallelPair …
                                             ⊢ Eq (f.toFun (s.ι.toFun x)) (g.toFun (s.ι.toFun x))
                                           -/
    ⟨{  toFun := fun x => ⟨s.ι.toFun x, by apply ContinuousHom.congr_fun s.condition⟩
                                           /-
                                             🎉 no goals
                                           -/
        monotone' := fun _ _ h => s.ι.monotone h
        map_ωSup' := fun x => Subtype.ext (s.ι.continuous x)
            /-
              X Y : ωCPO
              f g : Quiver.Hom X Y
              s : CategoryTheory.Limits.Fork f g
              ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := fun x => ⟨s.ι.toFun x, ⋯⟩, …
            -/
      }, by ext; rfl, fun hm => by
                 /-
                   🎉 no goals
                 -/
      /-
        X Y : ωCPO
        f g : Quiver.Hom X Y
        s : CategoryTheory.Limits.Fork f g
        m✝ : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingP …
        hm : Eq (CategoryTheory.CategoryStruct.comp m✝ (ωCPO.HasEqualizers.equalizer f …
        ⊢ Eq m✝ { toFun := fun x => ⟨s.ι.toFun x, ⋯⟩, monotone' := ⋯, map_ωSup' := ⋯ }
      -/
      apply ContinuousHom.ext _ _ fun x => Subtype.ext ?_ -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext`
      /-
        X Y : ωCPO
        f g : Quiver.Hom X Y
        s : CategoryTheory.Limits.Fork f g
        m✝ : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingP …
        hm : Eq (CategoryTheory.CategoryStruct.comp m✝ (ωCPO.HasEqualizers.equalizer f …
        x : ↑(((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingParallelPair …
        ⊢ Eq ↑(m✝ x) ↑({ toFun := fun x => ⟨s.ι.toFun x, ⋯⟩, monotone' := ⋯, map_ωSup' …
      -/
      apply ContinuousHom.congr_fun hm⟩
      /-
        🎉 no goals
      -/


instance : HasProducts.{v} ωCPO.{v} :=
  fun _ => { has_limit := fun _ => hasLimitOfIso Discrete.natIsoFunctor.symm }


instance {X Y : ωCPO.{v}} (f g : X ⟶ Y) : HasLimit (parallelPair f g) :=
  HasLimit.mk ⟨_, HasEqualizers.isEqualizer f g⟩


instance : HasEqualizers ωCPO.{v} :=
  hasEqualizers_of_hasLimit_parallelPair _


instance : HasLimits ωCPO.{v} :=
  has_limits_of_hasEqualizers_and_products


