/-- Bundled monotone (aka, increasing) function -/
structure OrderHom (α β : Type*) [Preorder α] [Preorder β] where
  /-- The underlying function of an `OrderHom`. -/
  toFun : α → β
  /-- The underlying function of an `OrderHom` is monotone. -/
  monotone' : Monotone toFun


/-- Notation for an `OrderHom`. -/
infixr:25 " →o " => OrderHom


/-- An order embedding is an embedding `f : α ↪ β` such that `a ≤ b ↔ (f a) ≤ (f b)`.
This definition is an abbreviation of `RelEmbedding (≤) (≤)`. -/
abbrev OrderEmbedding (α β : Type*) [LE α] [LE β] :=
  @RelEmbedding α β (· ≤ ·) (· ≤ ·)


/-- Notation for an `OrderEmbedding`. -/
infixl:25 " ↪o " => OrderEmbedding


/-- An order isomorphism is an equivalence such that `a ≤ b ↔ (f a) ≤ (f b)`.
This definition is an abbreviation of `RelIso (≤) (≤)`. -/
abbrev OrderIso (α β : Type*) [LE α] [LE β] :=
  @RelIso α β (· ≤ ·) (· ≤ ·)


/-- Notation for an `OrderIso`. -/
infixl:25 " ≃o " => OrderIso


/-- `OrderHomClass F α b` asserts that `F` is a type of `≤`-preserving morphisms. -/
abbrev OrderHomClass (F : Type*) (α β : outParam Type*) [LE α] [LE β] [FunLike F α β] :=
  RelHomClass F ((· ≤ ·) : α → α → Prop) ((· ≤ ·) : β → β → Prop)


/-- `OrderIsoClass F α β` states that `F` is a type of order isomorphisms.

You should extend this class when you extend `OrderIso`. -/
class OrderIsoClass (F : Type*) (α β : outParam Type*) [LE α] [LE β] [EquivLike F α β] :
    Prop where
  /-- An order isomorphism respects `≤`. -/
  map_le_map_iff (f : F) {a b : α} : f a ≤ f b ↔ a ≤ b


attribute [simp] map_le_map_iff


/-- Turn an element of a type `F` satisfying `OrderIsoClass F α β` into an actual
`OrderIso`. This is declared as the default coercion from `F` to `α ≃o β`. -/
@[coe]
def OrderIsoClass.toOrderIso [LE α] [LE β] [EquivLike F α β] [OrderIsoClass F α β] (f : F) :
    α ≃o β :=
  { EquivLike.toEquiv f with map_rel_iff' := map_le_map_iff f }


/-- Any type satisfying `OrderIsoClass` can be cast into `OrderIso` via
`OrderIsoClass.toOrderIso`. -/
instance [LE α] [LE β] [EquivLike F α β] [OrderIsoClass F α β] : CoeTC F (α ≃o β) :=
  ⟨OrderIsoClass.toOrderIso⟩

-- See note [lower instance priority]

instance (priority := 100) OrderIsoClass.toOrderHomClass [LE α] [LE β]
    [EquivLike F α β] [OrderIsoClass F α β] : OrderHomClass F α β :=
  { EquivLike.toEmbeddingLike (E := F) with
    map_rel := fun f _ _ => (map_le_map_iff f).2 }


protected theorem monotone (f : F) : Monotone f := fun _ _ => map_rel f


protected theorem mono (f : F) : Monotone f := fun _ _ => map_rel f


@[gcongr] protected lemma GCongr.mono (f : F) {a b : α} (hab : a ≤ b) : f a ≤ f b :=
  OrderHomClass.mono f hab


/-- Turn an element of a type `F` satisfying `OrderHomClass F α β` into an actual
`OrderHom`. This is declared as the default coercion from `F` to `α →o β`. -/
@[coe]
def toOrderHom (f : F) : α →o β where
  toFun := f
  monotone' := OrderHomClass.monotone f


/-- Any type satisfying `OrderHomClass` can be cast into `OrderHom` via
`OrderHomClass.toOrderHom`. -/
instance : CoeTC F (α →o β) :=
  ⟨toOrderHom⟩


@[simp]
theorem map_inv_le_iff (f : F) {a : α} {b : β} : EquivLike.inv f b ≤ a ↔ b ≤ f a := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : LE α
    inst✝² : LE β
    inst✝¹ : EquivLike F α β
    inst✝ : OrderIsoClass F α β
    f : F
    a : α
    b : β
    ⊢ Iff (LE.le (EquivLike.inv f b) a) (LE.le b (f a))
  -/
  convert (map_le_map_iff f (a := EquivLike.inv f b) (b := a)).symm
  /-
    case h.e'_2.h.e'_3
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : LE α
    inst✝² : LE β
    inst✝¹ : EquivLike F α β
    inst✝ : OrderIsoClass F α β
    f : F
    a : α
    b : β
    ⊢ Eq b (f (EquivLike.inv f b))
  -/
  exact (EquivLike.right_inv f _).symm
  /-
    🎉 no goals
  -/


theorem map_inv_le_map_inv_iff (f : F) {a b : β} :
    EquivLike.inv f b ≤ EquivLike.inv f a ↔ b ≤ a := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : LE α
    inst✝² : LE β
    inst✝¹ : EquivLike F α β
    inst✝ : OrderIsoClass F α β
    f : F
    a b : β
    ⊢ Iff (LE.le (EquivLike.inv f b) (EquivLike.inv f a)) (LE.le b a)
  -/
  simp
  /-
    🎉 no goals
  -/

-- Porting note: needed to add explicit arguments to map_le_map_iff

@[simp]
theorem le_map_inv_iff (f : F) {a : α} {b : β} : a ≤ EquivLike.inv f b ↔ f a ≤ b := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : LE α
    inst✝² : LE β
    inst✝¹ : EquivLike F α β
    inst✝ : OrderIsoClass F α β
    f : F
    a : α
    b : β
    ⊢ Iff (LE.le a (EquivLike.inv f b)) (LE.le (f a) b)
  -/
  convert (map_le_map_iff f (a := a) (b := EquivLike.inv f b)).symm
  /-
    case h.e'_2.h.e'_4
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : LE α
    inst✝² : LE β
    inst✝¹ : EquivLike F α β
    inst✝ : OrderIsoClass F α β
    f : F
    a : α
    b : β
    ⊢ Eq b (f (EquivLike.inv f b))
  -/
  exact (EquivLike.right_inv _ _).symm
  /-
    🎉 no goals
  -/


theorem map_lt_map_iff (f : F) {a b : α} : f a < f b ↔ a < b :=
  lt_iff_lt_of_le_iff_le' (map_le_map_iff f) (map_le_map_iff f)


@[simp]
theorem map_inv_lt_iff (f : F) {a : α} {b : β} : EquivLike.inv f b < a ↔ b < f a := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Preorder α
    inst✝² : Preorder β
    inst✝¹ : EquivLike F α β
    inst✝ : OrderIsoClass F α β
    f : F
    a : α
    b : β
    ⊢ Iff (LT.lt (EquivLike.inv f b) a) (LT.lt b (f a))
  -/
  rw [← map_lt_map_iff f]
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Preorder α
    inst✝² : Preorder β
    inst✝¹ : EquivLike F α β
    inst✝ : OrderIsoClass F α β
    f : F
    a : α
    b : β
    ⊢ Iff (LT.lt (f (EquivLike.inv f b)) (f a)) (LT.lt b (f a))
  -/
  simp only [EquivLike.apply_inv_apply]
  /-
    🎉 no goals
  -/


theorem map_inv_lt_map_inv_iff (f : F) {a b : β} :
    EquivLike.inv f b < EquivLike.inv f a ↔ b < a := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Preorder α
    inst✝² : Preorder β
    inst✝¹ : EquivLike F α β
    inst✝ : OrderIsoClass F α β
    f : F
    a b : β
    ⊢ Iff (LT.lt (EquivLike.inv f b) (EquivLike.inv f a)) (LT.lt b a)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem lt_map_inv_iff (f : F) {a : α} {b : β} : a < EquivLike.inv f b ↔ f a < b := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Preorder α
    inst✝² : Preorder β
    inst✝¹ : EquivLike F α β
    inst✝ : OrderIsoClass F α β
    f : F
    a : α
    b : β
    ⊢ Iff (LT.lt a (EquivLike.inv f b)) (LT.lt (f a) b)
  -/
  rw [← map_lt_map_iff f]
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Preorder α
    inst✝² : Preorder β
    inst✝¹ : EquivLike F α β
    inst✝ : OrderIsoClass F α β
    f : F
    a : α
    b : β
    ⊢ Iff (LT.lt (f a) (f (EquivLike.inv f b))) (LT.lt (f a) b)
  -/
  simp only [EquivLike.apply_inv_apply]
  /-
    🎉 no goals
  -/


instance : FunLike (α →o β) α β where
  coe := toFun
                             /-
                               F : Type u_1
                               α : Type u_2
                               β : Type u_3
                               γ : Type u_4
                               δ : Type u_5
                               inst✝³ : Preorder α
                               inst✝² : Preorder β
                               inst✝¹ : Preorder γ
                               inst✝ : Preorder δ
                               f g : OrderHom α β
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


instance : OrderHomClass (α →o β) α β where
  map_rel f _ _ h := f.monotone' h


@[simp] theorem coe_mk (f : α → β) (hf : Monotone f) : ⇑(mk f hf) = f := rfl


protected theorem monotone (f : α →o β) : Monotone f :=
  f.monotone'


protected theorem mono (f : α →o β) : Monotone f :=
  f.monotone


/-- See Note [custom simps projection]. We give this manually so that we use `toFun` as the
projection directly instead. -/
def Simps.coe (f : α →o β) : α → β := f

/- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: all other DFunLike classes use `apply` instead of `coe`
for the projection names. Maybe we should change this. -/

@[simp] theorem toFun_eq_coe (f : α →o β) : f.toFun = f := rfl

-- See library note [partially-applied ext lemmas]

@[ext]
theorem ext (f g : α →o β) (h : (f : α → β) = g) : f = g :=
  DFunLike.coe_injective h


@[simp] theorem coe_eq (f : α →o β) : OrderHomClass.toOrderHom f = f := rfl


@[simp] theorem _root_.OrderHomClass.coe_coe {F} [FunLike F α β] [OrderHomClass F α β] (f : F) :
    ⇑(f : α →o β) = f :=
  rfl


/-- One can lift an unbundled monotone function to a bundled one. -/
protected instance canLift : CanLift (α → β) (α →o β) (↑) Monotone where
  prf f h := ⟨⟨f, h⟩, rfl⟩


/-- Copy of an `OrderHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (f : α →o β) (f' : α → β) (h : f' = f) : α →o β :=
  ⟨f', h.symm.subst f.monotone'⟩


@[simp]
theorem coe_copy (f : α →o β) (f' : α → β) (h : f' = f) : (f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : α →o β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- The identity function as bundled monotone function. -/
@[simps (config := .asFn)]
def id : α →o α :=
  ⟨_root_.id, monotone_id⟩


instance : Inhabited (α →o α) :=
  ⟨id⟩


/-- The preorder structure of `α →o β` is pointwise inequality: `f ≤ g ↔ ∀ a, f a ≤ g a`. -/
instance : Preorder (α →o β) :=
  @Preorder.lift (α →o β) (α → β) _ toFun


instance {β : Type*} [PartialOrder β] : PartialOrder (α →o β) :=
  @PartialOrder.lift (α →o β) (α → β) _ toFun ext


theorem le_def {f g : α →o β} : f ≤ g ↔ ∀ x, f x ≤ g x :=
  Iff.rfl


@[simp, norm_cast]
theorem coe_le_coe {f g : α →o β} : (f : α → β) ≤ g ↔ f ≤ g :=
  Iff.rfl


@[simp]
theorem mk_le_mk {f g : α → β} {hf hg} : mk f hf ≤ mk g hg ↔ f ≤ g :=
  Iff.rfl


@[mono]
theorem apply_mono {f g : α →o β} {x y : α} (h₁ : f ≤ g) (h₂ : x ≤ y) : f x ≤ g y :=
  (h₁ x).trans <| g.mono h₂


/-- Curry/uncurry as an order isomorphism between `α × β →o γ` and `α →o β →o γ`. -/
def curry : (α × β →o γ) ≃o (α →o β →o γ) where
  toFun f := ⟨fun x ↦ ⟨Function.curry f x, fun _ _ h ↦ f.mono ⟨le_rfl, h⟩⟩, fun _ _ h _ =>
    f.mono ⟨h, le_rfl⟩⟩
  invFun f := ⟨Function.uncurry fun x ↦ f x, fun x y h ↦ (f.mono h.1 x.2).trans ((f y.1).mono h.2)⟩
  left_inv _ := rfl
  right_inv _ := rfl
                     /-
                       F : Type u_1
                       α : Type u_2
                       β : Type u_3
                       γ : Type u_4
                       δ : Type u_5
                       inst✝³ : Preorder α
                       inst✝² : Preorder β
                       inst✝¹ : Preorder γ
                       inst✝ : Preorder δ
                       ⊢ ∀ {a b : OrderHom (Prod α β) γ}, Iff (LE.le ({ toFun := fun f => { toFun :=  …
                     -/
  map_rel_iff' := by simp [le_def]
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem curry_apply (f : α × β →o γ) (x : α) (y : β) : curry f x y = f (x, y) :=
  rfl


@[simp]
theorem curry_symm_apply (f : α →o β →o γ) (x : α × β) : curry.symm f x = f x.1 x.2 :=
  rfl


/-- The composition of two bundled monotone functions. -/
@[simps (config := .asFn)]
def comp (g : β →o γ) (f : α →o β) : α →o γ :=
  ⟨g ∘ f, g.mono.comp f.mono⟩


@[mono]
theorem comp_mono ⦃g₁ g₂ : β →o γ⦄ (hg : g₁ ≤ g₂) ⦃f₁ f₂ : α →o β⦄ (hf : f₁ ≤ f₂) :
    g₁.comp f₁ ≤ g₂.comp f₂ := fun _ => (hg _).trans (g₂.mono <| hf _)


@[simp] lemma mk_comp_mk (g : β → γ) (f : α → β) (hg hf) :
    comp ⟨g, hg⟩ ⟨f, hf⟩ = ⟨g ∘ f, hg.comp hf⟩ := rfl


/-- The composition of two bundled monotone functions, a fully bundled version. -/
@[simps! (config := .asFn)]
def compₘ : (β →o γ) →o (α →o β) →o α →o γ :=
  curry ⟨fun f : (β →o γ) × (α →o β) => f.1.comp f.2, fun _ _ h => comp_mono h.1 h.2⟩


@[simp]
theorem comp_id (f : α →o β) : comp f id = f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderHom α β
    ⊢ Eq (f.comp OrderHom.id) f
  -/
  ext
  /-
    case h.h
    α : Type u_2
    β : Type u_3
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderHom α β
    x✝ : α
    ⊢ Eq ((f.comp OrderHom.id) x✝) (f x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem id_comp (f : α →o β) : comp id f = f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderHom α β
    ⊢ Eq (OrderHom.id.comp f) f
  -/
  ext
  /-
    case h.h
    α : Type u_2
    β : Type u_3
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderHom α β
    x✝ : α
    ⊢ Eq ((OrderHom.id.comp f) x✝) (f x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Constant function bundled as an `OrderHom`. -/
@[simps (config := .asFn)]
def const (α : Type*) [Preorder α] {β : Type*} [Preorder β] : β →o α →o β where
  toFun b := ⟨Function.const α b, fun _ _ _ => le_rfl⟩
  monotone' _ _ h _ := h


@[simp]
theorem const_comp (f : α →o β) (c : γ) : (const β c).comp f = const α c :=
  rfl


@[simp]
theorem comp_const (γ : Type*) [Preorder γ] (f : α →o β) (c : α) :
    f.comp (const γ c) = const γ (f c) :=
  rfl


/-- Given two bundled monotone maps `f`, `g`, `f.prod g` is the map `x ↦ (f x, g x)` bundled as a
`OrderHom`. -/
@[simps]
protected def prod (f : α →o β) (g : α →o γ) : α →o β × γ :=
  ⟨fun x => (f x, g x), fun _ _ h => ⟨f.mono h, g.mono h⟩⟩


@[mono]
theorem prod_mono {f₁ f₂ : α →o β} (hf : f₁ ≤ f₂) {g₁ g₂ : α →o γ} (hg : g₁ ≤ g₂) :
    f₁.prod g₁ ≤ f₂.prod g₂ := fun _ => Prod.le_def.2 ⟨hf _, hg _⟩


theorem comp_prod_comp_same (f₁ f₂ : β →o γ) (g : α →o β) :
    (f₁.comp g).prod (f₂.comp g) = (f₁.prod f₂).comp g :=
  rfl


/-- Given two bundled monotone maps `f`, `g`, `f.prod g` is the map `x ↦ (f x, g x)` bundled as a
`OrderHom`. This is a fully bundled version. -/
@[simps!]
def prodₘ : (α →o β) →o (α →o γ) →o α →o β × γ :=
  curry ⟨fun f : (α →o β) × (α →o γ) => f.1.prod f.2, fun _ _ h => prod_mono h.1 h.2⟩


/-- Diagonal embedding of `α` into `α × α` as an `OrderHom`. -/
@[simps!]
def diag : α →o α × α :=
  id.prod id


/-- Restriction of `f : α →o α →o β` to the diagonal. -/
@[simps! (config := { simpRhs := true })]
def onDiag (f : α →o α →o β) : α →o β :=
  (curry.symm f).comp diag


/-- `Prod.fst` as an `OrderHom`. -/
@[simps]
def fst : α × β →o α :=
  ⟨Prod.fst, fun _ _ h => h.1⟩


/-- `Prod.snd` as an `OrderHom`. -/
@[simps]
def snd : α × β →o β :=
  ⟨Prod.snd, fun _ _ h => h.2⟩


@[simp]
theorem fst_prod_snd : (fst : α × β →o α).prod snd = id := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    ⊢ Eq (OrderHom.fst.prod OrderHom.snd) OrderHom.id
  -/
  ext ⟨x, y⟩ : 2
  /-
    case h.h.mk
    α : Type u_2
    β : Type u_3
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    x : α
    y : β
    ⊢ Eq ((OrderHom.fst.prod OrderHom.snd) { fst := x, snd := y }) (OrderHom.id {  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem fst_comp_prod (f : α →o β) (g : α →o γ) : fst.comp (f.prod g) = f :=
  ext _ _ rfl


@[simp]
theorem snd_comp_prod (f : α →o β) (g : α →o γ) : snd.comp (f.prod g) = g :=
  ext _ _ rfl


/-- Order isomorphism between the space of monotone maps to `β × γ` and the product of the spaces
of monotone maps to `β` and `γ`. -/
@[simps]
def prodIso : (α →o β × γ) ≃o (α →o β) × (α →o γ) where
  toFun f := (fst.comp f, snd.comp f)
  invFun f := f.1.prod f.2
  left_inv _ := rfl
  right_inv _ := rfl
  map_rel_iff' := forall_and.symm


/-- `Prod.map` of two `OrderHom`s as an `OrderHom`. -/
@[simps]
def prodMap (f : α →o β) (g : γ →o δ) : α × γ →o β × δ :=
  ⟨Prod.map f g, fun _ _ h => ⟨f.mono h.1, g.mono h.2⟩⟩


/-- Evaluation of an unbundled function at a point (`Function.eval`) as an `OrderHom`. -/
@[simps (config := .asFn)]
def _root_.Pi.evalOrderHom (i : ι) : (∀ j, π j) →o π i :=
  ⟨Function.eval i, Function.monotone_eval i⟩


/-- The "forgetful functor" from `α →o β` to `α → β` that takes the underlying function,
is monotone. -/
@[simps (config := .asFn)]
def coeFnHom : (α →o β) →o α → β where
  toFun f := f
  monotone' _ _ h := h


/-- Function application `fun f => f a` (for fixed `a`) is a monotone function from the
monotone function space `α →o β` to `β`. See also `Pi.evalOrderHom`. -/
@[simps! (config := .asFn)]
def apply (x : α) : (α →o β) →o β :=
  (Pi.evalOrderHom x).comp coeFnHom


/-- Construct a bundled monotone map `α →o Π i, π i` from a family of monotone maps
`f i : α →o π i`. -/
@[simps]
def pi (f : ∀ i, α →o π i) : α →o ∀ i, π i :=
  ⟨fun x i => f i x, fun _ _ h i => (f i).mono h⟩


/-- Order isomorphism between bundled monotone maps `α →o Π i, π i` and families of bundled monotone
maps `Π i, α →o π i`. -/
@[simps]
def piIso : (α →o ∀ i, π i) ≃o ∀ i, α →o π i where
  toFun f i := (Pi.evalOrderHom i).comp f
  invFun := pi
  left_inv _ := rfl
  right_inv _ := rfl
  map_rel_iff' := forall_swap


/-- `Subtype.val` as a bundled monotone function. -/
@[simps (config := .asFn)]
def Subtype.val (p : α → Prop) : Subtype p →o α :=
  ⟨_root_.Subtype.val, fun _ _ h => h⟩


/-- `Subtype.impEmbedding` as an order embedding. -/
@[simps!]
def _root_.Subtype.orderEmbedding {p q : α → Prop} (h : ∀ a, p a → q a) :
    {x // p x} ↪o {x // q x} :=
  { Subtype.impEmbedding _ _ h with
                       /-
                         F : Type u_1
                         α : Type u_2
                         β : Type u_3
                         γ : Type u_4
                         δ : Type u_5
                         inst✝⁴ : Preorder α
                         inst✝³ : Preorder β
                         inst✝² : Preorder γ
                         inst✝¹ : Preorder δ
                         ι : Type u_6
                         π : ι → Type u_7
                         inst✝ : (i : ι) → Preorder (π i)
                         p q : α → Prop
                         h : ∀ (a : α), p a → q a
                         ⊢ ∀ {a b : Subtype fun x => p x}, Iff (LE.le (__src✝ a) (__src✝ b)) (LE.le a b)
                       -/
    map_rel_iff' := by aesop }
                       /-
                         🎉 no goals
                       -/


/-- There is a unique monotone map from a subsingleton to itself. -/
instance unique [Subsingleton α] : Unique (α →o α) where
  default := OrderHom.id
  uniq _ := ext _ _ (Subsingleton.elim _ _)


theorem orderHom_eq_id [Subsingleton α] (g : α →o α) : g = OrderHom.id :=
  Subsingleton.elim _ _


/-- Reinterpret a bundled monotone function as a monotone function between dual orders. -/
@[simps]
protected def dual : (α →o β) ≃ (αᵒᵈ →o βᵒᵈ) where
  toFun f := ⟨(OrderDual.toDual : β → βᵒᵈ) ∘ (f : α → β) ∘
    (OrderDual.ofDual : αᵒᵈ → α), f.mono.dual⟩
  invFun f := ⟨OrderDual.ofDual ∘ f ∘ OrderDual.toDual, f.mono.dual⟩
  left_inv _ := rfl
  right_inv _ := rfl


@[simp]
theorem dual_id : (OrderHom.id : α →o α).dual = OrderHom.id :=
  rfl


@[simp]
theorem dual_comp (g : β →o γ) (f : α →o β) :
    (g.comp f).dual = g.dual.comp f.dual :=
  rfl


@[simp]
theorem symm_dual_id : OrderHom.dual.symm OrderHom.id = (OrderHom.id : α →o α) :=
  rfl


@[simp]
theorem symm_dual_comp (g : βᵒᵈ →o γᵒᵈ) (f : αᵒᵈ →o βᵒᵈ) :
    OrderHom.dual.symm (g.comp f) = (OrderHom.dual.symm g).comp (OrderHom.dual.symm f) :=
  rfl


/-- `OrderHom.dual` as an order isomorphism. -/
def dualIso (α β : Type*) [Preorder α] [Preorder β] : (α →o β) ≃o (αᵒᵈ →o βᵒᵈ)ᵒᵈ where
  toEquiv := OrderHom.dual.trans OrderDual.toDual
  map_rel_iff' := Iff.rfl


/-- Lift an order homomorphism `f : α →o β` to an order homomorphism `WithBot α →o WithBot β`. -/
@[simps (config := .asFn)]
protected def withBotMap (f : α →o β) : WithBot α →o WithBot β :=
  ⟨WithBot.map f, f.mono.withBot_map⟩


/-- Lift an order homomorphism `f : α →o β` to an order homomorphism `WithTop α →o WithTop β`. -/
@[simps (config := .asFn)]
protected def withTopMap (f : α →o β) : WithTop α →o WithTop β :=
  ⟨WithTop.map f, f.mono.withTop_map⟩


/-- Lift an order homomorphism `f : α →o β` to an order homomorphism `ULift α →o ULift β` in a
higher universe. -/
@[simps!]
def uliftMap (f : α →o β) : ULift α →o ULift β :=
  ⟨fun i => ⟨f i.down⟩, fun _ _ h ↦ f.monotone h⟩


/-- Embeddings of partial orders that preserve `<` also preserve `≤`. -/
def RelEmbedding.orderEmbeddingOfLTEmbedding [PartialOrder α] [PartialOrder β]
    (f : ((· < ·) : α → α → Prop) ↪r ((· < ·) : β → β → Prop)) : α ↪o β :=
  { f with
    map_rel_iff' := by
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝¹ : PartialOrder α
        inst✝ : PartialOrder β
        f : RelEmbedding (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
        ⊢ ∀ {a b : α}, Iff (LE.le (f.toEmbedding a) (f.toEmbedding b)) (LE.le a b)
      -/
      intros
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝¹ : PartialOrder α
        inst✝ : PartialOrder β
        f : RelEmbedding (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
        a✝ b✝ : α
        ⊢ Iff (LE.le (f.toEmbedding a✝) (f.toEmbedding b✝)) (LE.le a✝ b✝)
      -/
      simp [le_iff_lt_or_eq, f.map_rel_iff, f.injective.eq_iff] }
      /-
        🎉 no goals
      -/


@[simp]
theorem RelEmbedding.orderEmbeddingOfLTEmbedding_apply [PartialOrder α] [PartialOrder β]
    {f : ((· < ·) : α → α → Prop) ↪r ((· < ·) : β → β → Prop)} {x : α} :
    RelEmbedding.orderEmbeddingOfLTEmbedding f x = f x :=
  rfl


/-- `<` is preserved by order embeddings of preorders. -/
def ltEmbedding : ((· < ·) : α → α → Prop) ↪r ((· < ·) : β → β → Prop) :=
                              /-
                                F : Type u_1
                                α : Type u_2
                                β : Type u_3
                                γ : Type u_4
                                δ : Type u_5
                                inst✝¹ : Preorder α
                                inst✝ : Preorder β
                                f : OrderEmbedding α β
                                ⊢ ∀ {a b : α}, Iff (LT.lt (f.toEmbedding a) (f.toEmbedding b)) (LT.lt a b)
                              -/
  { f with map_rel_iff' := by intros; simp [lt_iff_le_not_le, f.map_rel_iff] }
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem ltEmbedding_apply (x : α) : f.ltEmbedding x = f x :=
  rfl


@[simp]
theorem le_iff_le {a b} : f a ≤ f b ↔ a ≤ b :=
  f.map_rel_iff


@[simp]
theorem lt_iff_lt {a b} : f a < f b ↔ a < b :=
  f.ltEmbedding.map_rel_iff


theorem eq_iff_eq {a b} : f a = f b ↔ a = b :=
  f.injective.eq_iff


protected theorem monotone : Monotone f :=
  OrderHomClass.monotone f


protected theorem strictMono : StrictMono f := fun _ _ => f.lt_iff_lt.2


protected theorem acc (a : α) : Acc (· < ·) (f a) → Acc (· < ·) a :=
  f.ltEmbedding.acc a


protected theorem wellFounded (f : α ↪o β) :
    WellFounded ((· < ·) : β → β → Prop) → WellFounded ((· < ·) : α → α → Prop) :=
  f.ltEmbedding.wellFounded


protected theorem isWellOrder [IsWellOrder β (· < ·)] (f : α ↪o β) : IsWellOrder α (· < ·) :=
  f.ltEmbedding.isWellOrder


/-- An order embedding is also an order embedding between dual orders. -/
protected def dual : αᵒᵈ ↪o βᵒᵈ :=
  ⟨f.toEmbedding, f.map_rel_iff⟩


/-- A preorder which embeds into a well-founded preorder is itself well-founded. -/
protected theorem wellFoundedLT [WellFoundedLT β] (f : α ↪o β) : WellFoundedLT α where
  wf := f.wellFounded IsWellFounded.wf


/-- A preorder which embeds into a preorder in which `(· > ·)` is well-founded
also has `(· > ·)` well-founded. -/
protected theorem wellFoundedGT [WellFoundedGT β] (f : α ↪o β) : WellFoundedGT α :=
  @OrderEmbedding.wellFoundedLT αᵒᵈ _ _ _ _ f.dual


/-- A version of `WithBot.map` for order embeddings. -/
@[simps (config := .asFn)]
protected def withBotMap (f : α ↪o β) : WithBot α ↪o WithBot β :=
  { f.toEmbedding.optionMap with
    toFun := WithBot.map f,
    map_rel_iff' := @fun a b => WithBot.map_le_iff f f.map_rel_iff a b }


/-- A version of `WithTop.map` for order embeddings. -/
@[simps (config := .asFn)]
protected def withTopMap (f : α ↪o β) : WithTop α ↪o WithTop β :=
  { f.dual.withBotMap.dual with toFun := WithTop.map f }


/-- Coercion `α → WithBot α` as an `OrderEmbedding`. -/
@[simps (config := .asFn)]
protected def withBotCoe : α ↪o WithBot α where
  toFun := .some
  inj' := Option.some_injective _
  map_rel_iff' := WithBot.coe_le_coe


/-- Coercion `α → WithTop α` as an `OrderEmbedding`. -/
@[simps (config := .asFn)]
protected def withTopCoe : α ↪o WithTop α :=
  { (OrderEmbedding.withBotCoe (α := αᵒᵈ)).dual with toFun := .some }


/-- To define an order embedding from a partial order to a preorder it suffices to give a function
together with a proof that it satisfies `f a ≤ f b ↔ a ≤ b`.
-/
def ofMapLEIff {α β} [PartialOrder α] [Preorder β] (f : α → β) (hf : ∀ a b, f a ≤ f b ↔ a ≤ b) :
    α ↪o β :=
  RelEmbedding.ofMapRelIff f hf


@[simp]
theorem coe_ofMapLEIff {α β} [PartialOrder α] [Preorder β] {f : α → β} (h) :
    ⇑(ofMapLEIff f h) = f :=
  rfl


/-- A strictly monotone map from a linear order is an order embedding. -/
def ofStrictMono {α β} [LinearOrder α] [Preorder β] (f : α → β) (h : StrictMono f) : α ↪o β :=
  ofMapLEIff f fun _ _ => h.le_iff_le


@[simp]
theorem coe_ofStrictMono {α β} [LinearOrder α] [Preorder β] {f : α → β} (h : StrictMono f) :
    ⇑(ofStrictMono f h) = f :=
  rfl


/-- Embedding of a subtype into the ambient type as an `OrderEmbedding`. -/
@[simps! (config := .asFn)]
def subtype (p : α → Prop) : Subtype p ↪o α :=
  ⟨Function.Embedding.subtype p, Iff.rfl⟩


/-- Convert an `OrderEmbedding` to an `OrderHom`. -/
@[simps (config := .asFn)]
def toOrderHom {X Y : Type*} [Preorder X] [Preorder Y] (f : X ↪o Y) : X →o Y where
  toFun := f
  monotone' := f.monotone


/-- The trivial embedding from an empty preorder to another preorder -/
@[simps] def ofIsEmpty [IsEmpty α] : α ↪o β where
  toFun := isEmptyElim
  inj' := isEmptyElim
  map_rel_iff' {a} := isEmptyElim a


@[simp, norm_cast]
lemma coe_ofIsEmpty [IsEmpty α] : (ofIsEmpty : α ↪o β) = (isEmptyElim : α → β) := rfl


/-- If the images by an order embedding of two elements are disjoint,
then they are themselves disjoint. -/
lemma Disjoint.of_orderEmbedding [OrderBot α] [OrderBot β] {a₁ a₂ : α} :
    Disjoint (f a₁) (f a₂) → Disjoint a₁ a₂ := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : PartialOrder α
    inst✝² : PartialOrder β
    f : OrderEmbedding α β
    inst✝¹ : OrderBot α
    inst✝ : OrderBot β
    a₁ a₂ : α
    ⊢ Disjoint (f a₁) (f a₂) → Disjoint a₁ a₂
  -/
  intro h x h₁ h₂
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : PartialOrder α
    inst✝² : PartialOrder β
    f : OrderEmbedding α β
    inst✝¹ : OrderBot α
    inst✝ : OrderBot β
    a₁ a₂ : α
    h : Disjoint (f a₁) (f a₂)
    x : α
    h₁ : LE.le x a₁
    h₂ : LE.le x a₂
    ⊢ LE.le x Bot.bot
  -/
  rw [← f.le_iff_le] at h₁ h₂ ⊢
  calc
    f x ≤ ⊥ := h h₁ h₂
    _ ≤ f ⊥ := bot_le


/-- If the images by an order embedding of two elements are codisjoint,
then they are themselves codisjoint. -/
lemma Codisjoint.of_orderEmbedding [OrderTop α] [OrderTop β] {a₁ a₂ : α} :
    Codisjoint (f a₁) (f a₂) → Codisjoint a₁ a₂ :=
  Disjoint.of_orderEmbedding (α := αᵒᵈ) (β := βᵒᵈ) f.dual


/-- If the images by an order embedding of two elements are complements,
then they are themselves complements. -/
lemma IsCompl.of_orderEmbedding [BoundedOrder α] [BoundedOrder β] {a₁ a₂ : α} :
    IsCompl (f a₁) (f a₂) → IsCompl a₁ a₂ := fun ⟨hd, hcd⟩ ↦
  ⟨Disjoint.of_orderEmbedding f hd, Codisjoint.of_orderEmbedding f hcd⟩


/-- A bundled expression of the fact that a map between partial orders that is strictly monotone
is weakly monotone. -/
@[simps (config := .asFn)]
def toOrderHom : α →o β where
  toFun := f
  monotone' := StrictMono.monotone fun _ _ => f.map_rel


theorem RelEmbedding.toOrderHom_injective
    (f : ((· < ·) : α → α → Prop) ↪r ((· < ·) : β → β → Prop)) :
    Function.Injective (f : ((· < ·) : α → α → Prop) →r ((· < ·) : β → β → Prop)).toOrderHom :=
  fun _ _ h => f.injective h


instance : EquivLike (α ≃o β) α β where
  coe f := f.toFun
  inv f := f.invFun
  left_inv f := f.left_inv
  right_inv f := f.right_inv
  coe_injective' f g h₁ h₂ := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝² : LE α
      inst✝¹ : LE β
      inst✝ : LE γ
      f g : OrderIso α β
      h₁ : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      h₂ : Eq ((fun f => f.invFun) f) ((fun f => f.invFun) g)
      ⊢ Eq f g
    -/
    obtain ⟨⟨_, _⟩, _⟩ := f
    /-
      case mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝² : LE α
      inst✝¹ : LE β
      inst✝ : LE γ
      g : OrderIso α β
      toFun✝ : α → β
      invFun✝ : β → α
      left_inv✝ : Function.LeftInverse invFun✝ toFun✝
      right_inv✝ : Function.RightInverse invFun✝ toFun✝
      map_rel_iff'✝ : ∀ {a b : α}, Iff (LE.le ({ toFun := toFun✝, invFun := invFun✝, …
      h₁ : Eq ((fun f => f.toFun) { toFun := toFun✝, invFun := invFun✝, left_inv :=  …
      h₂ : Eq ((fun f => f.invFun) { toFun := toFun✝, invFun := invFun✝, left_inv := …
      ⊢ Eq { toFun := toFun✝, invFun := invFun✝, left_inv := left_inv✝, right_inv := …
    -/
    obtain ⟨⟨_, _⟩, _⟩ := g
    /-
      case mk.mk.mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝² : LE α
      inst✝¹ : LE β
      inst✝ : LE γ
      toFun✝¹ : α → β
      invFun✝¹ : β → α
      left_inv✝¹ : Function.LeftInverse invFun✝¹ toFun✝¹
      right_inv✝¹ : Function.RightInverse invFun✝¹ toFun✝¹
      map_rel_iff'✝¹ : ∀ {a b : α}, Iff (LE.le ({ toFun := toFun✝¹, invFun := invFun …
      toFun✝ : α → β
      invFun✝ : β → α
      left_inv✝ : Function.LeftInverse invFun✝ toFun✝
      right_inv✝ : Function.RightInverse invFun✝ toFun✝
      map_rel_iff'✝ : ∀ {a b : α}, Iff (LE.le ({ toFun := toFun✝, invFun := invFun✝, …
      h₁ : Eq ((fun f => f.toFun) { toFun := toFun✝¹, invFun := invFun✝¹, left_inv : …
      h₂ : Eq ((fun f => f.invFun) { toFun := toFun✝¹, invFun := invFun✝¹, left_inv  …
      ⊢ Eq { toFun := toFun✝¹, invFun := invFun✝¹, left_inv := left_inv✝¹, right_inv …
    -/
    congr
    /-
      🎉 no goals
    -/


instance : OrderIsoClass (α ≃o β) α β where
  map_le_map_iff f _ _ := f.map_rel_iff'


@[simp]
theorem toFun_eq_coe {f : α ≃o β} : f.toFun = f :=
  rfl

-- See note [partially-applied ext lemmas]

@[ext]
theorem ext {f g : α ≃o β} (h : (f : α → β) = g) : f = g :=
  DFunLike.coe_injective h


/-- Reinterpret an order isomorphism as an order embedding. -/
def toOrderEmbedding (e : α ≃o β) : α ↪o β :=
  e.toRelEmbedding


@[simp]
theorem coe_toOrderEmbedding (e : α ≃o β) : ⇑e.toOrderEmbedding = e :=
  rfl


protected theorem bijective (e : α ≃o β) : Function.Bijective e :=
  e.toEquiv.bijective


protected theorem injective (e : α ≃o β) : Function.Injective e :=
  e.toEquiv.injective


protected theorem surjective (e : α ≃o β) : Function.Surjective e :=
  e.toEquiv.surjective


theorem apply_eq_iff_eq (e : α ≃o β) {x y : α} : e x = e y ↔ x = y :=
  e.toEquiv.apply_eq_iff_eq


/-- Identity order isomorphism. -/
def refl (α : Type*) [LE α] : α ≃o α :=
  RelIso.refl (· ≤ ·)


@[simp]
theorem coe_refl : ⇑(refl α) = id :=
  rfl


@[simp]
theorem refl_apply (x : α) : refl α x = x :=
  rfl


@[simp]
theorem refl_toEquiv : (refl α).toEquiv = Equiv.refl α :=
  rfl


/-- Inverse of an order isomorphism. -/
def symm (e : α ≃o β) : β ≃o α := RelIso.symm e


@[simp]
theorem apply_symm_apply (e : α ≃o β) (x : β) : e (e.symm x) = x :=
  e.toEquiv.apply_symm_apply x


@[simp]
theorem symm_apply_apply (e : α ≃o β) (x : α) : e.symm (e x) = x :=
  e.toEquiv.symm_apply_apply x


@[simp]
theorem symm_refl (α : Type*) [LE α] : (refl α).symm = refl α :=
  rfl


theorem apply_eq_iff_eq_symm_apply (e : α ≃o β) (x : α) (y : β) : e x = y ↔ x = e.symm y :=
  e.toEquiv.apply_eq_iff_eq_symm_apply


theorem symm_apply_eq (e : α ≃o β) {x : α} {y : β} : e.symm y = x ↔ y = e x :=
  e.toEquiv.symm_apply_eq


@[simp]
theorem symm_symm (e : α ≃o β) : e.symm.symm = e := rfl


theorem symm_bijective : Function.Bijective (OrderIso.symm : (α ≃o β) → β ≃o α) :=
  Function.bijective_iff_has_inverse.mpr ⟨_, symm_symm, symm_symm⟩


theorem symm_injective : Function.Injective (symm : α ≃o β → β ≃o α) :=
  symm_bijective.injective


@[simp]
theorem toEquiv_symm (e : α ≃o β) : e.toEquiv.symm = e.symm.toEquiv :=
  rfl


/-- Composition of two order isomorphisms is an order isomorphism. -/
@[trans]
def trans (e : α ≃o β) (e' : β ≃o γ) : α ≃o γ :=
  RelIso.trans e e'


@[simp]
theorem coe_trans (e : α ≃o β) (e' : β ≃o γ) : ⇑(e.trans e') = e' ∘ e :=
  rfl


@[simp]
theorem trans_apply (e : α ≃o β) (e' : β ≃o γ) (x : α) : e.trans e' x = e' (e x) :=
  rfl


@[simp]
theorem refl_trans (e : α ≃o β) : (refl α).trans e = e := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : LE α
    inst✝ : LE β
    e : OrderIso α β
    ⊢ Eq ((OrderIso.refl α).trans e) e
  -/
  ext x
  /-
    case h.h
    α : Type u_2
    β : Type u_3
    inst✝¹ : LE α
    inst✝ : LE β
    e : OrderIso α β
    x : α
    ⊢ Eq (((OrderIso.refl α).trans e) x) (e x)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem trans_refl (e : α ≃o β) : e.trans (refl β) = e := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : LE α
    inst✝ : LE β
    e : OrderIso α β
    ⊢ Eq (e.trans (OrderIso.refl β)) e
  -/
  ext x
  /-
    case h.h
    α : Type u_2
    β : Type u_3
    inst✝¹ : LE α
    inst✝ : LE β
    e : OrderIso α β
    x : α
    ⊢ Eq ((e.trans (OrderIso.refl β)) x) (e x)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem symm_trans_apply (e₁ : α ≃o β) (e₂ : β ≃o γ) (c : γ) :
    (e₁.trans e₂).symm c = e₁.symm (e₂.symm c) :=
  rfl


theorem symm_trans (e₁ : α ≃o β) (e₂ : β ≃o γ) : (e₁.trans e₂).symm = e₂.symm.trans e₁.symm :=
  rfl


@[simp]
theorem self_trans_symm (e : α ≃o β) : e.trans e.symm = OrderIso.refl α :=
  RelIso.self_trans_symm e


@[simp]
theorem symm_trans_self (e : α ≃o β) : e.symm.trans e = OrderIso.refl β :=
  RelIso.symm_trans_self e


/-- An order isomorphism between the domains and codomains of two prosets of
order homomorphisms gives an order isomorphism between the two function prosets. -/
@[simps apply symm_apply]
def arrowCongr {α β γ δ} [Preorder α] [Preorder β] [Preorder γ] [Preorder δ]
    (f : α ≃o γ) (g : β ≃o δ) : (α →o β) ≃o (γ →o δ) where
  toFun  p := .comp g <| .comp p f.symm
  invFun p := .comp g.symm <| .comp p f
  left_inv p := DFunLike.coe_injective <| by
    /-
      F : Type u_1
      α✝ : Type u_2
      β✝ : Type u_3
      γ✝ : Type u_4
      δ✝ : Type u_5
      inst✝⁶ : LE α✝
      inst✝⁵ : LE β✝
      inst✝⁴ : LE γ✝
      α : Type ?u.66201
      β : Type ?u.66204
      γ : Type ?u.66207
      δ : Type ?u.66210
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : Preorder γ
      inst✝ : Preorder δ
      f : OrderIso α γ
      g : OrderIso β δ
      p : OrderHom α β
      ⊢ Eq ((fun f => ⇑f) ((fun p => (↑g.symm).comp (p.comp ↑f)) ((fun p => (↑g).com …
    -/
    change (g.symm ∘ g) ∘ p ∘ (f.symm ∘ f) = p
    simp only [← DFunLike.coe_eq_coe_fn, ← OrderIso.coe_trans, Function.id_comp,
               OrderIso.self_trans_symm, OrderIso.coe_refl, Function.comp_id]
  right_inv p := DFunLike.coe_injective <| by
    /-
      F : Type u_1
      α✝ : Type u_2
      β✝ : Type u_3
      γ✝ : Type u_4
      δ✝ : Type u_5
      inst✝⁶ : LE α✝
      inst✝⁵ : LE β✝
      inst✝⁴ : LE γ✝
      α : Type ?u.66201
      β : Type ?u.66204
      γ : Type ?u.66207
      δ : Type ?u.66210
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : Preorder γ
      inst✝ : Preorder δ
      f : OrderIso α γ
      g : OrderIso β δ
      p : OrderHom γ δ
      ⊢ Eq ((fun f => ⇑f) ((fun p => (↑g).comp (p.comp ↑f.symm)) ((fun p => (↑g.symm …
    -/
    change (g ∘ g.symm) ∘ p ∘ (f ∘ f.symm) = p
    simp only [← DFunLike.coe_eq_coe_fn, ← OrderIso.coe_trans, Function.id_comp,
               OrderIso.symm_trans_self, OrderIso.coe_refl, Function.comp_id]
  map_rel_iff' {p q} := by
    simp only [Equiv.coe_fn_mk, OrderHom.le_def, OrderHom.comp_coe,
               OrderHomClass.coe_coe, Function.comp_apply, map_le_map_iff]
    /-
      F : Type u_1
      α✝ : Type u_2
      β✝ : Type u_3
      γ✝ : Type u_4
      δ✝ : Type u_5
      inst✝⁶ : LE α✝
      inst✝⁵ : LE β✝
      inst✝⁴ : LE γ✝
      α : Type ?u.66201
      β : Type ?u.66204
      γ : Type ?u.66207
      δ : Type ?u.66210
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : Preorder γ
      inst✝ : Preorder δ
      f : OrderIso α γ
      g : OrderIso β δ
      p q : OrderHom α β
      ⊢ Iff (∀ (x : γ), LE.le (p (f.symm x)) (q (f.symm x))) (∀ (x : α), LE.le (p x) …
    -/
    exact Iff.symm f.forall_congr_left
    /-
      🎉 no goals
    -/


/-- If `α` and `β` are order-isomorphic then the two orders of order-homomorphisms
from `α` and `β` to themselves are order-isomorphic. -/
@[simps! apply symm_apply]
def conj {α β} [Preorder α] [Preorder β] (f : α ≃o β) : (α →o α) ≃ (β →o β) :=
  arrowCongr f f


/-- `Prod.swap` as an `OrderIso`. -/
def prodComm : α × β ≃o β × α where
  toEquiv := Equiv.prodComm α β
  map_rel_iff' := Prod.swap_le_swap


@[simp]
theorem coe_prodComm : ⇑(prodComm : α × β ≃o β × α) = Prod.swap :=
  rfl


@[simp]
theorem prodComm_symm : (prodComm : α × β ≃o β × α).symm = prodComm :=
  rfl


/-- The order isomorphism between a type and its double dual. -/
def dualDual : α ≃o αᵒᵈᵒᵈ :=
  refl α


@[simp]
theorem coe_dualDual : ⇑(dualDual α) = toDual ∘ toDual :=
  rfl


@[simp]
theorem coe_dualDual_symm : ⇑(dualDual α).symm = ofDual ∘ ofDual :=
  rfl


@[simp]
theorem dualDual_apply (a : α) : dualDual α a = toDual (toDual a) :=
  rfl


@[simp]
theorem dualDual_symm_apply (a : αᵒᵈᵒᵈ) : (dualDual α).symm a = ofDual (ofDual a) :=
  rfl


theorem le_iff_le (e : α ≃o β) {x y : α} : e x ≤ e y ↔ x ≤ y :=
  e.map_rel_iff


@[gcongr] protected alias ⟨_, GCongr.orderIso_apply_le_apply⟩ := le_iff_le


theorem le_symm_apply (e : α ≃o β) {x : α} {y : β} : x ≤ e.symm y ↔ e x ≤ y :=
  e.rel_symm_apply


theorem symm_apply_le (e : α ≃o β) {x : α} {y : β} : e.symm y ≤ x ↔ y ≤ e x :=
  e.symm_apply_rel


protected theorem monotone (e : α ≃o β) : Monotone e :=
  e.toOrderEmbedding.monotone


protected theorem strictMono (e : α ≃o β) : StrictMono e :=
  e.toOrderEmbedding.strictMono


@[simp]
theorem lt_iff_lt (e : α ≃o β) {x y : α} : e x < e y ↔ x < y :=
  e.toOrderEmbedding.lt_iff_lt


@[gcongr] protected alias ⟨_, GCongr.orderIso_apply_lt_apply⟩ := lt_iff_lt


/-- Converts an `OrderIso` into a `RelIso (<) (<)`. -/
def toRelIsoLT (e : α ≃o β) : ((· < ·) : α → α → Prop) ≃r ((· < ·) : β → β → Prop) :=
  ⟨e.toEquiv, lt_iff_lt e⟩


@[simp]
theorem toRelIsoLT_apply (e : α ≃o β) (x : α) : e.toRelIsoLT x = e x :=
  rfl


@[simp]
theorem toRelIsoLT_symm (e : α ≃o β) : e.toRelIsoLT.symm = e.symm.toRelIsoLT :=
  rfl


/-- Converts a `RelIso (<) (<)` into an `OrderIso`. -/
def ofRelIsoLT {α β} [PartialOrder α] [PartialOrder β]
    (e : ((· < ·) : α → α → Prop) ≃r ((· < ·) : β → β → Prop)) : α ≃o β :=
                 /-
                   F : Type u_1
                   α✝ : Type u_2
                   β✝ : Type u_3
                   γ : Type u_4
                   δ : Type u_5
                   inst✝³ : Preorder α✝
                   inst✝² : Preorder β✝
                   α : Type ?u.78674
                   β : Type ?u.78677
                   inst✝¹ : PartialOrder α
                   inst✝ : PartialOrder β
                   e : RelIso (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
                   ⊢ ∀ {a b : α}, Iff (LE.le (e.toEquiv a) (e.toEquiv b)) (LE.le a b)
                 -/
  ⟨e.toEquiv, by simp [le_iff_eq_or_lt, e.map_rel_iff, e.injective.eq_iff]⟩
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem ofRelIsoLT_apply {α β} [PartialOrder α] [PartialOrder β]
    (e : ((· < ·) : α → α → Prop) ≃r ((· < ·) : β → β → Prop)) (x : α) : ofRelIsoLT e x = e x :=
  rfl


@[simp]
theorem ofRelIsoLT_symm {α β} [PartialOrder α] [PartialOrder β]
    (e : ((· < ·) : α → α → Prop) ≃r ((· < ·) : β → β → Prop)) :
    (ofRelIsoLT e).symm = ofRelIsoLT e.symm :=
  rfl


@[simp]
theorem ofRelIsoLT_toRelIsoLT {α β} [PartialOrder α] [PartialOrder β] (e : α ≃o β) :
    ofRelIsoLT (toRelIsoLT e) = e := by
  /-
    α : Type u_6
    β : Type u_7
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    e : OrderIso α β
    ⊢ Eq (OrderIso.ofRelIsoLT e.toRelIsoLT) e
  -/
  ext
  /-
    case h.h
    α : Type u_6
    β : Type u_7
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    e : OrderIso α β
    x✝ : α
    ⊢ Eq ((OrderIso.ofRelIsoLT e.toRelIsoLT) x✝) (e x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem toRelIsoLT_ofRelIsoLT {α β} [PartialOrder α] [PartialOrder β]
    (e : ((· < ·) : α → α → Prop) ≃r ((· < ·) : β → β → Prop)) : toRelIsoLT (ofRelIsoLT e) = e := by
  /-
    α : Type u_6
    β : Type u_7
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    e : RelIso (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
    ⊢ Eq (OrderIso.ofRelIsoLT e).toRelIsoLT e
  -/
  ext
  /-
    case h
    α : Type u_6
    β : Type u_7
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    e : RelIso (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
    x✝ : α
    ⊢ Eq ((OrderIso.ofRelIsoLT e).toRelIsoLT x✝) (e x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- To show that `f : α → β`, `g : β → α` make up an order isomorphism of linear orders,
    it suffices to prove `cmp a (g b) = cmp (f a) b`. -/
def ofCmpEqCmp {α β} [LinearOrder α] [LinearOrder β] (f : α → β) (g : β → α)
    (h : ∀ (a : α) (b : β), cmp a (g b) = cmp (f a) b) : α ≃o β :=
  have gf : ∀ a : α, a = g (f a) := by
    /-
      F : Type u_1
      α✝ : Type u_2
      β✝ : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝³ : Preorder α✝
      inst✝² : Preorder β✝
      α : Type ?u.81507
      β : Type ?u.81510
      inst✝¹ : LinearOrder α
      inst✝ : LinearOrder β
      f : α → β
      g : β → α
      h : ∀ (a : α) (b : β), Eq (cmp a (g b)) (cmp (f a) b)
      ⊢ ∀ (a : α), Eq a (g (f a))
    -/
    intro
    /-
      F : Type u_1
      α✝ : Type u_2
      β✝ : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝³ : Preorder α✝
      inst✝² : Preorder β✝
      α : Type ?u.81507
      β : Type ?u.81510
      inst✝¹ : LinearOrder α
      inst✝ : LinearOrder β
      f : α → β
      g : β → α
      h : ∀ (a : α) (b : β), Eq (cmp a (g b)) (cmp (f a) b)
      a✝ : α
      ⊢ Eq a✝ (g (f a✝))
    -/
    rw [← cmp_eq_eq_iff, h, cmp_self_eq_eq]
    /-
      🎉 no goals
    -/
  { toFun := f, invFun := g, left_inv := fun a => (gf a).symm,
    right_inv := by
      /-
        F : Type u_1
        α✝ : Type u_2
        β✝ : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : Preorder α✝
        inst✝² : Preorder β✝
        α : Type ?u.81507
        β : Type ?u.81510
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : α → β
        g : β → α
        h : ∀ (a : α) (b : β), Eq (cmp a (g b)) (cmp (f a) b)
        gf : ∀ (a : α), Eq a (g (f a))
        ⊢ Function.RightInverse g f
      -/
      intro
      /-
        F : Type u_1
        α✝ : Type u_2
        β✝ : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : Preorder α✝
        inst✝² : Preorder β✝
        α : Type ?u.81507
        β : Type ?u.81510
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : α → β
        g : β → α
        h : ∀ (a : α) (b : β), Eq (cmp a (g b)) (cmp (f a) b)
        gf : ∀ (a : α), Eq a (g (f a))
        x✝ : β
        ⊢ Eq (f (g x✝)) x✝
      -/
      rw [← cmp_eq_eq_iff, ← h, cmp_self_eq_eq],
      /-
        🎉 no goals
      -/
    map_rel_iff' := by
      /-
        F : Type u_1
        α✝ : Type u_2
        β✝ : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : Preorder α✝
        inst✝² : Preorder β✝
        α : Type ?u.81507
        β : Type ?u.81510
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : α → β
        g : β → α
        h : ∀ (a : α) (b : β), Eq (cmp a (g b)) (cmp (f a) b)
        gf : ∀ (a : α), Eq a (g (f a))
        ⊢ ∀ {a b : α}, Iff (LE.le ({ toFun := f, invFun := g, left_inv := ⋯, right_inv …
      -/
      intros a b
      /-
        F : Type u_1
        α✝ : Type u_2
        β✝ : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : Preorder α✝
        inst✝² : Preorder β✝
        α : Type ?u.81507
        β : Type ?u.81510
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : α → β
        g : β → α
        h : ∀ (a : α) (b : β), Eq (cmp a (g b)) (cmp (f a) b)
        gf : ∀ (a : α), Eq a (g (f a))
        a b : α
        ⊢ Iff (LE.le ({ toFun := f, invFun := g, left_inv := ⋯, right_inv := ⋯ } a) ({ …
      -/
      apply le_iff_le_of_cmp_eq_cmp
      /-
        case h
        F : Type u_1
        α✝ : Type u_2
        β✝ : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : Preorder α✝
        inst✝² : Preorder β✝
        α : Type ?u.81507
        β : Type ?u.81510
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : α → β
        g : β → α
        h : ∀ (a : α) (b : β), Eq (cmp a (g b)) (cmp (f a) b)
        gf : ∀ (a : α), Eq a (g (f a))
        a b : α
        ⊢ Eq (cmp ({ toFun := f, invFun := g, left_inv := ⋯, right_inv := ⋯ } a) ({ to …
      -/
      convert (h a (f b)).symm
      /-
        case h.e'_3.h.e'_5
        F : Type u_1
        α✝ : Type u_2
        β✝ : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : Preorder α✝
        inst✝² : Preorder β✝
        α : Type ?u.81507
        β : Type ?u.81510
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : α → β
        g : β → α
        h : ∀ (a : α) (b : β), Eq (cmp a (g b)) (cmp (f a) b)
        gf : ∀ (a : α), Eq a (g (f a))
        a b : α
        ⊢ Eq b (g (f b))
      -/
      apply gf }
      /-
        🎉 no goals
      -/


/-- To show that `f : α →o β` and `g : β →o α` make up an order isomorphism it is enough to show
    that `g` is the inverse of `f`-/
def ofHomInv {F G : Type*} [FunLike F α β] [OrderHomClass F α β] [FunLike G β α]
    [OrderHomClass G β α] (f : F) (g : G)
    (h₁ : (f : α →o β).comp (g : β →o α) = OrderHom.id)
    (h₂ : (g : β →o α).comp (f : α →o β) = OrderHom.id) :
    α ≃o β where
  toFun := f
  invFun := g
  left_inv := DFunLike.congr_fun h₂
  right_inv := DFunLike.congr_fun h₁
  map_rel_iff' := @fun a b =>
    ⟨fun h => by
      /-
        F✝ : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝⁵ : Preorder α
        inst✝⁴ : Preorder β
        F : Type u_6
        G : Type u_7
        inst✝³ : FunLike F α β
        inst✝² : OrderHomClass F α β
        inst✝¹ : FunLike G β α
        inst✝ : OrderHomClass G β α
        f : F
        g : G
        h₁ : Eq ((↑f).comp ↑g) OrderHom.id
        h₂ : Eq ((↑g).comp ↑f) OrderHom.id
        a b : α
        h : LE.le ({ toFun := ⇑f, invFun := ⇑g, left_inv := ⋯, right_inv := ⋯ } a) ({  …
        ⊢ LE.le a b
      -/
      replace h := map_rel g h
      rwa [Equiv.coe_fn_mk, show g (f a) = (g : β →o α).comp (f : α →o β) a from rfl,
        show g (f b) = (g : β →o α).comp (f : α →o β) b from rfl, h₂] at h,
      fun h => (f : α →o β).monotone h⟩


/-- Order isomorphism between `α → β` and `β`, where `α` has a unique element. -/
@[simps! toEquiv apply]
def funUnique (α β : Type*) [Unique α] [Preorder β] : (α → β) ≃o β where
  toEquiv := Equiv.funUnique α β
                     /-
                       F : Type u_1
                       α✝ : Type u_2
                       β✝ : Type u_3
                       γ : Type u_4
                       δ : Type u_5
                       inst✝³ : Preorder α✝
                       inst✝² : Preorder β✝
                       α : Type u_6
                       β : Type u_7
                       inst✝¹ : Unique α
                       inst✝ : Preorder β
                       ⊢ ∀ {a b : α → β}, Iff (LE.le ((Equiv.funUnique α β) a) ((Equiv.funUnique α β) …
                     -/
  map_rel_iff' := by simp [Pi.le_def, Unique.forall_iff]
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem funUnique_symm_apply {α β : Type*} [Unique α] [Preorder β] :
    ((funUnique α β).symm : β → α → β) = Function.const α :=
  rfl


/-- If `e` is an equivalence with monotone forward and inverse maps, then `e` is an
order isomorphism. -/
def toOrderIso (e : α ≃ β) (h₁ : Monotone e) (h₂ : Monotone e.symm) : α ≃o β :=
                   /-
                     F : Type u_1
                     α : Type u_2
                     β : Type u_3
                     γ : Type u_4
                     δ : Type u_5
                     inst✝¹ : Preorder α
                     inst✝ : Preorder β
                     e : Equiv α β
                     h₁ : Monotone ⇑e
                     h₂ : Monotone ⇑e.symm
                     a✝ b✝ : α
                     h : LE.le (e a✝) (e b✝)
                     ⊢ LE.le a✝ b✝
                   -/
  ⟨e, ⟨fun h => by simpa only [e.symm_apply_apply] using h₂ h, fun h => h₁ h⟩⟩
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem coe_toOrderIso (e : α ≃ β) (h₁ : Monotone e) (h₂ : Monotone e.symm) :
    ⇑(e.toOrderIso h₁ h₂) = e :=
  rfl


@[simp]
theorem toOrderIso_toEquiv (e : α ≃ β) (h₁ : Monotone e) (h₂ : Monotone e.symm) :
    (e.toOrderIso h₁ h₂).toEquiv = e :=
  rfl


/-- A strictly monotone function with a right inverse is an order isomorphism. -/
@[simps (config := .asFn)]
def orderIsoOfRightInverse (g : β → α) (hg : Function.RightInverse g f) : α ≃o β :=
  { OrderEmbedding.ofStrictMono f h_mono with
    toFun := f,
    invFun := g,
    left_inv := fun _ => h_mono.injective <| hg _,
    right_inv := hg }


/-- An order isomorphism is also an order isomorphism between dual orders. -/
protected def OrderIso.dual [LE α] [LE β] (f : α ≃o β) : αᵒᵈ ≃o βᵒᵈ :=
  ⟨f.toEquiv, f.map_rel_iff⟩


theorem OrderIso.map_bot' [LE α] [PartialOrder β] (f : α ≃o β) {x : α} {y : β} (hx : ∀ x', x ≤ x')
    (hy : ∀ y', y ≤ y') : f x = y := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : LE α
    inst✝ : PartialOrder β
    f : OrderIso α β
    x : α
    y : β
    hx : ∀ (x' : α), LE.le x x'
    hy : ∀ (y' : β), LE.le y y'
    ⊢ Eq (f x) y
  -/
  refine le_antisymm ?_ (hy _)
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : LE α
    inst✝ : PartialOrder β
    f : OrderIso α β
    x : α
    y : β
    hx : ∀ (x' : α), LE.le x x'
    hy : ∀ (y' : β), LE.le y y'
    ⊢ LE.le (f x) y
  -/
  rw [← f.apply_symm_apply y, f.map_rel_iff]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : LE α
    inst✝ : PartialOrder β
    f : OrderIso α β
    x : α
    y : β
    hx : ∀ (x' : α), LE.le x x'
    hy : ∀ (y' : β), LE.le y y'
    ⊢ LE.le x (f.symm y)
  -/
  apply hx
  /-
    🎉 no goals
  -/


theorem OrderIso.map_bot [LE α] [PartialOrder β] [OrderBot α] [OrderBot β] (f : α ≃o β) : f ⊥ = ⊥ :=
  f.map_bot' (fun _ => bot_le) fun _ => bot_le


theorem OrderIso.map_top' [LE α] [PartialOrder β] (f : α ≃o β) {x : α} {y : β} (hx : ∀ x', x' ≤ x)
    (hy : ∀ y', y' ≤ y) : f x = y :=
  f.dual.map_bot' hx hy


theorem OrderIso.map_top [LE α] [PartialOrder β] [OrderTop α] [OrderTop β] (f : α ≃o β) : f ⊤ = ⊤ :=
  f.dual.map_bot


theorem OrderEmbedding.map_inf_le [SemilatticeInf α] [SemilatticeInf β] (f : α ↪o β) (x y : α) :
    f (x ⊓ y) ≤ f x ⊓ f y :=
  f.monotone.map_inf_le x y


theorem OrderEmbedding.le_map_sup [SemilatticeSup α] [SemilatticeSup β] (f : α ↪o β) (x y : α) :
    f x ⊔ f y ≤ f (x ⊔ y) :=
  f.monotone.le_map_sup x y


theorem OrderIso.map_inf [SemilatticeInf α] [SemilatticeInf β] (f : α ≃o β) (x y : α) :
    f (x ⊓ y) = f x ⊓ f y := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeInf α
    inst✝ : SemilatticeInf β
    f : OrderIso α β
    x y : α
    ⊢ Eq (f (Min.min x y)) (Min.min (f x) (f y))
  -/
  refine (f.toOrderEmbedding.map_inf_le x y).antisymm ?_
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeInf α
    inst✝ : SemilatticeInf β
    f : OrderIso α β
    x y : α
    ⊢ LE.le (Min.min (f.toOrderEmbedding x) (f.toOrderEmbedding y)) (f.toOrderEmbe …
  -/
  apply f.symm.le_iff_le.1
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeInf α
    inst✝ : SemilatticeInf β
    f : OrderIso α β
    x y : α
    ⊢ LE.le (f.symm (Min.min (f.toOrderEmbedding x) (f.toOrderEmbedding y))) (f.sy …
  -/
  simpa using f.symm.toOrderEmbedding.map_inf_le (f x) (f y)
  /-
    🎉 no goals
  -/


theorem OrderIso.map_sup [SemilatticeSup α] [SemilatticeSup β] (f : α ≃o β) (x y : α) :
    f (x ⊔ y) = f x ⊔ f y :=
  f.dual.map_inf x y


theorem OrderIso.isMax_apply {α β : Type*} [Preorder α] [Preorder β] (f : α ≃o β) {x : α} :
    IsMax (f x) ↔ IsMax x := by
  /-
    α : Type u_6
    β : Type u_7
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    x : α
    ⊢ Iff (IsMax (f x)) (IsMax x)
  -/
  refine ⟨f.strictMono.isMax_of_apply, ?_⟩
  /-
    α : Type u_6
    β : Type u_7
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    x : α
    ⊢ IsMax x → IsMax (f x)
  -/
  conv_lhs => rw [← f.symm_apply_apply x]
  /-
    α : Type u_6
    β : Type u_7
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    x : α
    ⊢ IsMax (f.symm (f x)) → IsMax (f x)
  -/
  exact f.symm.strictMono.isMax_of_apply
  /-
    🎉 no goals
  -/


theorem OrderIso.isMin_apply {α β : Type*} [Preorder α] [Preorder β] (f : α ≃o β) {x : α} :
    IsMin (f x) ↔ IsMin x := by
  /-
    α : Type u_6
    β : Type u_7
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    x : α
    ⊢ Iff (IsMin (f x)) (IsMin x)
  -/
  refine ⟨f.strictMono.isMin_of_apply, ?_⟩
  /-
    α : Type u_6
    β : Type u_7
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    x : α
    ⊢ IsMin x → IsMin (f x)
  -/
  conv_lhs => rw [← f.symm_apply_apply x]
  /-
    α : Type u_6
    β : Type u_7
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    x : α
    ⊢ IsMin (f.symm (f x)) → IsMin (f x)
  -/
  exact f.symm.strictMono.isMin_of_apply
  /-
    🎉 no goals
  -/


/-- Note that this goal could also be stated `(Disjoint on f) a b` -/
theorem Disjoint.map_orderIso [SemilatticeInf α] [OrderBot α] [SemilatticeInf β] [OrderBot β]
    {a b : α} (f : α ≃o β) (ha : Disjoint a b) : Disjoint (f a) (f b) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : SemilatticeInf α
    inst✝² : OrderBot α
    inst✝¹ : SemilatticeInf β
    inst✝ : OrderBot β
    a b : α
    f : OrderIso α β
    ha : Disjoint a b
    ⊢ Disjoint (f a) (f b)
  -/
  rw [disjoint_iff_inf_le, ← f.map_inf, ← f.map_bot]
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : SemilatticeInf α
    inst✝² : OrderBot α
    inst✝¹ : SemilatticeInf β
    inst✝ : OrderBot β
    a b : α
    f : OrderIso α β
    ha : Disjoint a b
    ⊢ LE.le (f (Min.min a b)) (f Bot.bot)
  -/
  exact f.monotone ha.le_bot
  /-
    🎉 no goals
  -/


/-- Note that this goal could also be stated `(Codisjoint on f) a b` -/
theorem Codisjoint.map_orderIso [SemilatticeSup α] [OrderTop α] [SemilatticeSup β] [OrderTop β]
    {a b : α} (f : α ≃o β) (ha : Codisjoint a b) : Codisjoint (f a) (f b) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : SemilatticeSup α
    inst✝² : OrderTop α
    inst✝¹ : SemilatticeSup β
    inst✝ : OrderTop β
    a b : α
    f : OrderIso α β
    ha : Codisjoint a b
    ⊢ Codisjoint (f a) (f b)
  -/
  rw [codisjoint_iff_le_sup, ← f.map_sup, ← f.map_top]
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : SemilatticeSup α
    inst✝² : OrderTop α
    inst✝¹ : SemilatticeSup β
    inst✝ : OrderTop β
    a b : α
    f : OrderIso α β
    ha : Codisjoint a b
    ⊢ LE.le (f Top.top) (f (Max.max a b))
  -/
  exact f.monotone ha.top_le
  /-
    🎉 no goals
  -/


@[simp]
theorem disjoint_map_orderIso_iff [SemilatticeInf α] [OrderBot α] [SemilatticeInf β] [OrderBot β]
    {a b : α} (f : α ≃o β) : Disjoint (f a) (f b) ↔ Disjoint a b :=
  ⟨fun h => f.symm_apply_apply a ▸ f.symm_apply_apply b ▸ h.map_orderIso f.symm,
   fun h => h.map_orderIso f⟩


@[simp]
theorem codisjoint_map_orderIso_iff [SemilatticeSup α] [OrderTop α] [SemilatticeSup β] [OrderTop β]
    {a b : α} (f : α ≃o β) : Codisjoint (f a) (f b) ↔ Codisjoint a b :=
  ⟨fun h => f.symm_apply_apply a ▸ f.symm_apply_apply b ▸ h.map_orderIso f.symm,
   fun h => h.map_orderIso f⟩


/-- Taking the dual then adding `⊥` is the same as adding `⊤` then taking the dual.
This is the order iso form of `WithBot.ofDual`, as proven by `coe_toDualTopEquiv_eq`.
-/
protected def toDualTopEquiv [LE α] : WithBot αᵒᵈ ≃o (WithTop α)ᵒᵈ :=
  OrderIso.refl _


@[simp]
theorem toDualTopEquiv_coe [LE α] (a : α) :
    WithBot.toDualTopEquiv ↑(toDual a) = toDual (a : WithTop α) :=
  rfl


@[simp]
theorem toDualTopEquiv_symm_coe [LE α] (a : α) :
    WithBot.toDualTopEquiv.symm (toDual (a : WithTop α)) = ↑(toDual a) :=
  rfl


@[simp]
theorem toDualTopEquiv_bot [LE α] : WithBot.toDualTopEquiv (⊥ : WithBot αᵒᵈ) = ⊥ :=
  rfl


@[simp]
theorem toDualTopEquiv_symm_bot [LE α] : WithBot.toDualTopEquiv.symm (⊥ : (WithTop α)ᵒᵈ) = ⊥ :=
  rfl


theorem coe_toDualTopEquiv_eq [LE α] :
    (WithBot.toDualTopEquiv : WithBot αᵒᵈ → (WithTop α)ᵒᵈ) = toDual ∘ WithBot.ofDual :=
  funext fun _ => rfl


/-- The coercion `α → WithBot α` bundled as monotone map. -/
@[simps]
def coeOrderHom {α : Type*} [Preorder α] : α ↪o WithBot α where
  toFun := (↑)
  inj' := WithBot.coe_injective
  map_rel_iff' := WithBot.coe_le_coe


/-- Taking the dual then adding `⊤` is the same as adding `⊥` then taking the dual.
This is the order iso form of `WithTop.ofDual`, as proven by `coe_toDualBotEquiv_eq`. -/
protected def toDualBotEquiv [LE α] : WithTop αᵒᵈ ≃o (WithBot α)ᵒᵈ :=
  OrderIso.refl _


@[simp]
theorem toDualBotEquiv_coe [LE α] (a : α) :
    WithTop.toDualBotEquiv ↑(toDual a) = toDual (a : WithBot α) :=
  rfl


@[simp]
theorem toDualBotEquiv_symm_coe [LE α] (a : α) :
    WithTop.toDualBotEquiv.symm (toDual (a : WithBot α)) = ↑(toDual a) :=
  rfl


@[simp]
theorem toDualBotEquiv_top [LE α] : WithTop.toDualBotEquiv (⊤ : WithTop αᵒᵈ) = ⊤ :=
  rfl


@[simp]
theorem toDualBotEquiv_symm_top [LE α] : WithTop.toDualBotEquiv.symm (⊤ : (WithBot α)ᵒᵈ) = ⊤ :=
  rfl


theorem coe_toDualBotEquiv [LE α] :
    (WithTop.toDualBotEquiv : WithTop αᵒᵈ → (WithBot α)ᵒᵈ) = toDual ∘ WithTop.ofDual :=
  funext fun _ => rfl


/-- The coercion `α → WithTop α` bundled as monotone map. -/
@[simps]
def coeOrderHom {α : Type*} [Preorder α] : α ↪o WithTop α where
  toFun := (↑)
  inj' := WithTop.coe_injective
  map_rel_iff' := WithTop.coe_le_coe


/-- A version of `Equiv.optionCongr` for `WithTop`. -/
@[simps! apply]
def withTopCongr (e : α ≃o β) : WithTop α ≃o WithTop β :=
  { e.toOrderEmbedding.withTopMap with
    toEquiv := e.toEquiv.optionCongr }


@[simp]
theorem withTopCongr_refl : (OrderIso.refl α).withTopCongr = OrderIso.refl _ :=
  RelIso.toEquiv_injective Equiv.optionCongr_refl


@[simp]
theorem withTopCongr_symm (e : α ≃o β) : e.withTopCongr.symm = e.symm.withTopCongr :=
  RelIso.toEquiv_injective e.toEquiv.optionCongr_symm


@[simp]
theorem withTopCongr_trans (e₁ : α ≃o β) (e₂ : β ≃o γ) :
    e₁.withTopCongr.trans e₂.withTopCongr = (e₁.trans e₂).withTopCongr :=
  RelIso.toEquiv_injective <| e₁.toEquiv.optionCongr_trans e₂.toEquiv


/-- A version of `Equiv.optionCongr` for `WithBot`. -/
@[simps! apply]
def withBotCongr (e : α ≃o β) : WithBot α ≃o WithBot β :=
  { e.toOrderEmbedding.withBotMap with toEquiv := e.toEquiv.optionCongr }


@[simp]
theorem withBotCongr_refl : (OrderIso.refl α).withBotCongr = OrderIso.refl _ :=
  RelIso.toEquiv_injective Equiv.optionCongr_refl


@[simp]
theorem withBotCongr_symm (e : α ≃o β) : e.withBotCongr.symm = e.symm.withBotCongr :=
  RelIso.toEquiv_injective e.toEquiv.optionCongr_symm


@[simp]
theorem withBotCongr_trans (e₁ : α ≃o β) (e₂ : β ≃o γ) :
    e₁.withBotCongr.trans e₂.withBotCongr = (e₁.trans e₂).withBotCongr :=
  RelIso.toEquiv_injective <| e₁.toEquiv.optionCongr_trans e₂.toEquiv


theorem OrderIso.isCompl {x y : α} (h : IsCompl x y) : IsCompl (f x) (f y) :=
  ⟨h.1.map_orderIso _, h.2.map_orderIso _⟩


theorem OrderIso.isCompl_iff {x y : α} : IsCompl x y ↔ IsCompl (f x) (f y) :=
  ⟨f.isCompl, fun h => f.symm_apply_apply x ▸ f.symm_apply_apply y ▸ f.symm.isCompl h⟩


theorem OrderIso.complementedLattice [ComplementedLattice α] (f : α ≃o β) : ComplementedLattice β :=
  ⟨fun x => by
    /-
      α : Type u_2
      β : Type u_3
      inst✝⁴ : Lattice α
      inst✝³ : Lattice β
      inst✝² : BoundedOrder α
      inst✝¹ : BoundedOrder β
      inst✝ : ComplementedLattice α
      f : OrderIso α β
      x : β
      ⊢ Exists fun b => IsCompl x b
    -/
    obtain ⟨y, hy⟩ := exists_isCompl (f.symm x)
    /-
      case intro
      α : Type u_2
      β : Type u_3
      inst✝⁴ : Lattice α
      inst✝³ : Lattice β
      inst✝² : BoundedOrder α
      inst✝¹ : BoundedOrder β
      inst✝ : ComplementedLattice α
      f : OrderIso α β
      x : β
      y : α
      hy : IsCompl (f.symm x) y
      ⊢ Exists fun b => IsCompl x b
    -/
    rw [← f.symm_apply_apply y] at hy
    /-
      case intro
      α : Type u_2
      β : Type u_3
      inst✝⁴ : Lattice α
      inst✝³ : Lattice β
      inst✝² : BoundedOrder α
      inst✝¹ : BoundedOrder β
      inst✝ : ComplementedLattice α
      f : OrderIso α β
      x : β
      y : α
      hy : IsCompl (f.symm x) (f.symm (f y))
      ⊢ Exists fun b => IsCompl x b
    -/
    exact ⟨f y, f.symm.isCompl_iff.2 hy⟩⟩
    /-
      🎉 no goals
    -/


theorem OrderIso.complementedLattice_iff (f : α ≃o β) :
    ComplementedLattice α ↔ ComplementedLattice β :=
      /-
        α : Type u_2
        β : Type u_3
        inst✝³ : Lattice α
        inst✝² : Lattice β
        inst✝¹ : BoundedOrder α
        inst✝ : BoundedOrder β
        f : OrderIso α β
        ⊢ ComplementedLattice α → ComplementedLattice β
      -/
  ⟨by intro; exact f.complementedLattice,
             /-
               🎉 no goals
             -/
      /-
        α : Type u_2
        β : Type u_3
        inst✝³ : Lattice α
        inst✝² : Lattice β
        inst✝¹ : BoundedOrder α
        inst✝ : BoundedOrder β
        f : OrderIso α β
        ⊢ ComplementedLattice β → ComplementedLattice α
      -/
   by intro; exact f.symm.complementedLattice⟩
             /-
               🎉 no goals
             -/


lemma denselyOrdered_iff_of_orderIsoClass {X Y F : Type*} [Preorder X] [Preorder Y]
    [EquivLike F X Y] [OrderIsoClass F X Y] (f : F) :
    DenselyOrdered X ↔ DenselyOrdered Y := by
  /-
    X : Type u_6
    Y : Type u_7
    F : Type u_8
    inst✝³ : Preorder X
    inst✝² : Preorder Y
    inst✝¹ : EquivLike F X Y
    inst✝ : OrderIsoClass F X Y
    f : F
    ⊢ Iff (DenselyOrdered X) (DenselyOrdered Y)
  -/
  constructor
    /-
      case mp
      X : Type u_6
      Y : Type u_7
      F : Type u_8
      inst✝³ : Preorder X
      inst✝² : Preorder Y
      inst✝¹ : EquivLike F X Y
      inst✝ : OrderIsoClass F X Y
      f : F
      ⊢ DenselyOrdered X → DenselyOrdered Y
    -/
  · intro H
    /-
      case mp
      X : Type u_6
      Y : Type u_7
      F : Type u_8
      inst✝³ : Preorder X
      inst✝² : Preorder Y
      inst✝¹ : EquivLike F X Y
      inst✝ : OrderIsoClass F X Y
      f : F
      H : DenselyOrdered X
      ⊢ DenselyOrdered Y
    -/
    refine ⟨fun a b h ↦ ?_⟩
    /-
      case mp
      X : Type u_6
      Y : Type u_7
      F : Type u_8
      inst✝³ : Preorder X
      inst✝² : Preorder Y
      inst✝¹ : EquivLike F X Y
      inst✝ : OrderIsoClass F X Y
      f : F
      H : DenselyOrdered X
      a b : Y
      h : LT.lt a b
      ⊢ Exists fun a_1 => And (LT.lt a a_1) (LT.lt a_1 b)
    -/
    obtain ⟨c, hc⟩ := exists_between ((map_inv_lt_map_inv_iff f).mpr h)
    /-
      case mp.intro
      X : Type u_6
      Y : Type u_7
      F : Type u_8
      inst✝³ : Preorder X
      inst✝² : Preorder Y
      inst✝¹ : EquivLike F X Y
      inst✝ : OrderIsoClass F X Y
      f : F
      H : DenselyOrdered X
      a b : Y
      h : LT.lt a b
      c : X
      hc : And (LT.lt (EquivLike.inv f a) c) (LT.lt c (EquivLike.inv f b))
      ⊢ Exists fun a_1 => And (LT.lt a a_1) (LT.lt a_1 b)
    -/
    exact ⟨f c, by simpa using hc⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u_6
      Y : Type u_7
      F : Type u_8
      inst✝³ : Preorder X
      inst✝² : Preorder Y
      inst✝¹ : EquivLike F X Y
      inst✝ : OrderIsoClass F X Y
      f : F
      ⊢ DenselyOrdered Y → DenselyOrdered X
    -/
  · intro H
    /-
      case mpr
      X : Type u_6
      Y : Type u_7
      F : Type u_8
      inst✝³ : Preorder X
      inst✝² : Preorder Y
      inst✝¹ : EquivLike F X Y
      inst✝ : OrderIsoClass F X Y
      f : F
      H : DenselyOrdered Y
      ⊢ DenselyOrdered X
    -/
    refine ⟨fun a b h ↦ ?_⟩
    /-
      case mpr
      X : Type u_6
      Y : Type u_7
      F : Type u_8
      inst✝³ : Preorder X
      inst✝² : Preorder Y
      inst✝¹ : EquivLike F X Y
      inst✝ : OrderIsoClass F X Y
      f : F
      H : DenselyOrdered Y
      a b : X
      h : LT.lt a b
      ⊢ Exists fun a_1 => And (LT.lt a a_1) (LT.lt a_1 b)
    -/
    obtain ⟨c, hc⟩ := exists_between ((map_lt_map_iff f).mpr h)
    /-
      case mpr.intro
      X : Type u_6
      Y : Type u_7
      F : Type u_8
      inst✝³ : Preorder X
      inst✝² : Preorder Y
      inst✝¹ : EquivLike F X Y
      inst✝ : OrderIsoClass F X Y
      f : F
      H : DenselyOrdered Y
      a b : X
      h : LT.lt a b
      c : Y
      hc : And (LT.lt (f a) c) (LT.lt c (f b))
      ⊢ Exists fun a_1 => And (LT.lt a a_1) (LT.lt a_1 b)
    -/
    exact ⟨EquivLike.inv f c, by simpa using hc⟩
    /-
      🎉 no goals
    -/


