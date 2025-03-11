/-- A homology data for a short complex consists of two compatible left and
right homology data -/
structure HomologyData where
  /-- a left homology data -/
  left : S.LeftHomologyData
  /-- a right homology data -/
  right : S.RightHomologyData
  /-- the compatibility isomorphism relating the two dual notions of
    `LeftHomologyData` and `RightHomologyData`  -/
  iso : left.H ≅ right.H
  /-- the pentagon relation expressing the compatibility of the left
  and right homology data -/
  comm : left.π ≫ iso.hom ≫ right.ι = left.i ≫ right.p := by aesop_cat


attribute [reassoc (attr := simp)] HomologyData.comm


/-- A homology map data for a morphism `φ : S₁ ⟶ S₂` where both `S₁` and `S₂` are
equipped with homology data consists of left and right homology map data. -/
structure HomologyMapData where
  /-- a left homology map data -/
  left : LeftHomologyMapData φ h₁.left h₂.left
  /-- a right homology map data -/
  right : RightHomologyMapData φ h₁.right h₂.right


@[reassoc]
lemma comm (h : HomologyMapData φ h₁ h₂) :
    h.left.φH ≫ h₂.iso.hom = h₁.iso.hom ≫ h.right.φH := by
  simp only [← cancel_epi h₁.left.π, ← cancel_mono h₂.right.ι, assoc,
    LeftHomologyMapData.commπ_assoc, HomologyData.comm, LeftHomologyMapData.commi_assoc,
    RightHomologyMapData.commι, HomologyData.comm_assoc, RightHomologyMapData.commp]


instance : Subsingleton (HomologyMapData φ h₁ h₂) := ⟨by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.HomologyData
    h₂ : S₂.HomologyData
    ⊢ ∀ (a b : CategoryTheory.ShortComplex.HomologyMapData φ h₁ h₂), Eq a b
  -/
  rintro ⟨left₁, right₁⟩ ⟨left₂, right₂⟩
  /-
    case mk.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.HomologyData
    h₂ : S₂.HomologyData
    left₁ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁.left h₂.left
    right₁ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁.right h₂.right
    left₂ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁.left h₂.left
    right₂ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁.right h₂.right
    ⊢ Eq { left := left₁, right := right₁ } { left := left₂, right := right₂ }
  -/
  simp only [mk.injEq, eq_iff_true_of_subsingleton, and_self]⟩
  /-
    🎉 no goals
  -/


instance : Inhabited (HomologyMapData φ h₁ h₂) :=
  ⟨⟨default, default⟩⟩


instance : Unique (HomologyMapData φ h₁ h₂) := Unique.mk' _


/-- A choice of the (unique) homology map data associated with a morphism
`φ : S₁ ⟶ S₂` where both short complexes `S₁` and `S₂` are equipped with
homology data. -/
def homologyMapData : HomologyMapData φ h₁ h₂ := default


lemma congr_left_φH {γ₁ γ₂ : HomologyMapData φ h₁ h₂} (eq : γ₁ = γ₂) :
                                  /-
                                    C : Type u
                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                    S₁ S₂ : CategoryTheory.ShortComplex C
                                    φ : Quiver.Hom S₁ S₂
                                    h₁ : S₁.HomologyData
                                    h₂ : S₂.HomologyData
                                    γ₁ γ₂ : CategoryTheory.ShortComplex.HomologyMapData φ h₁ h₂
                                    eq : Eq γ₁ γ₂
                                    ⊢ Eq γ₁.left.φH γ₂.left.φH
                                  -/
    γ₁.left.φH = γ₂.left.φH := by rw [eq]
                                  /-
                                    🎉 no goals
                                  -/


/-- When the first map `S.f` is zero, this is the homology data on `S` given
by any limit kernel fork of `S.g` -/
@[simps]
def ofIsLimitKernelFork (hf : S.f = 0) (c : KernelFork S.g) (hc : IsLimit c) :
    S.HomologyData where
  left := LeftHomologyData.ofIsLimitKernelFork S hf c hc
  right := RightHomologyData.ofIsLimitKernelFork S hf c hc
  iso := Iso.refl _


/-- When the first map `S.f` is zero, this is the homology data on `S` given
by the chosen `kernel S.g` -/
@[simps]
noncomputable def ofHasKernel (hf : S.f = 0) [HasKernel S.g] :
    S.HomologyData where
  left := LeftHomologyData.ofHasKernel S hf
  right := RightHomologyData.ofHasKernel S hf
  iso := Iso.refl _


/-- When the second map `S.g` is zero, this is the homology data on `S` given
by any colimit cokernel cofork of `S.f` -/
@[simps]
def ofIsColimitCokernelCofork (hg : S.g = 0) (c : CokernelCofork S.f) (hc : IsColimit c) :
    S.HomologyData where
  left := LeftHomologyData.ofIsColimitCokernelCofork S hg c hc
  right := RightHomologyData.ofIsColimitCokernelCofork S hg c hc
  iso := Iso.refl _


/-- When the second map `S.g` is zero, this is the homology data on `S` given by
the chosen `cokernel S.f` -/
@[simps]
noncomputable def ofHasCokernel (hg : S.g = 0) [HasCokernel S.f] :
    S.HomologyData where
  left := LeftHomologyData.ofHasCokernel S hg
  right := RightHomologyData.ofHasCokernel S hg
  iso := Iso.refl _


/-- When both `S.f` and `S.g` are zero, the middle object `S.X₂` gives a homology data on S -/
@[simps]
noncomputable def ofZeros (hf : S.f = 0) (hg : S.g = 0) :
    S.HomologyData where
  left := LeftHomologyData.ofZeros S hf hg
  right := RightHomologyData.ofZeros S hf hg
  iso := Iso.refl _


/-- If `φ : S₁ ⟶ S₂` is a morphism of short complexes such that `φ.τ₁` is epi, `φ.τ₂` is an iso
and `φ.τ₃` is mono, then a homology data for `S₁` induces a homology data for `S₂`.
The inverse construction is `ofEpiOfIsIsoOfMono'`. -/
@[simps]
noncomputable def ofEpiOfIsIsoOfMono (φ : S₁ ⟶ S₂) (h : HomologyData S₁)
    [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] : HomologyData S₂ where
  left := LeftHomologyData.ofEpiOfIsIsoOfMono φ h.left
  right := RightHomologyData.ofEpiOfIsIsoOfMono φ h.right
  iso := h.iso


/-- If `φ : S₁ ⟶ S₂` is a morphism of short complexes such that `φ.τ₁` is epi, `φ.τ₂` is an iso
and `φ.τ₃` is mono, then a homology data for `S₂` induces a homology data for `S₁`.
The inverse construction is `ofEpiOfIsIsoOfMono`. -/
@[simps]
noncomputable def ofEpiOfIsIsoOfMono' (φ : S₁ ⟶ S₂) (h : HomologyData S₂)
    [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] : HomologyData S₁ where
  left := LeftHomologyData.ofEpiOfIsIsoOfMono' φ h.left
  right := RightHomologyData.ofEpiOfIsIsoOfMono' φ h.right
  iso := h.iso


/-- If `e : S₁ ≅ S₂` is an isomorphism of short complexes and `h₁ : HomologyData S₁`,
this is the homology data for `S₂` deduced from the isomorphism. -/
@[simps!]
noncomputable def ofIso (e : S₁ ≅ S₂) (h : HomologyData S₁) :=
  h.ofEpiOfIsIsoOfMono e.hom


/-- A homology data for a short complex `S` induces a homology data for `S.op`. -/
@[simps]
def op (h : S.HomologyData) : S.op.HomologyData where
  left := h.right.op
  right := h.left.op
  iso := h.iso.op
                                  /-
                                    C : Type u
                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                    S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
                                    φ : Quiver.Hom S₁ S₂
                                    h₁ : S₁.HomologyData
                                    h₂ : S₂.HomologyData
                                    h : S.HomologyData
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.right.op.π (CategoryTheory.Category …
                                  -/
  comm := Quiver.Hom.unop_inj (by simp)
                                  /-
                                    🎉 no goals
                                  -/


/-- A homology data for a short complex `S` in the opposite category
induces a homology data for `S.unop`. -/
@[simps]
def unop {S : ShortComplex Cᵒᵖ} (h : S.HomologyData) : S.unop.HomologyData where
  left := h.right.unop
  right := h.left.unop
  iso := h.iso.unop
                                /-
                                  C : Type u
                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                  inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                  S✝ S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
                                  φ : Quiver.Hom S₁ S₂
                                  h₁ : S₁.HomologyData
                                  h₂ : S₂.HomologyData
                                  S : CategoryTheory.ShortComplex (Opposite C)
                                  h : S.HomologyData
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp h.right.unop.π (CategoryTheory.Catego …
                                -/
  comm := Quiver.Hom.op_inj (by simp)
                                /-
                                  🎉 no goals
                                -/


/-- A short complex `S` has homology when there exists a `S.HomologyData` -/
class HasHomology : Prop where
  /-- the condition that there exists a homology data -/
  condition : Nonempty S.HomologyData


/-- A chosen `S.HomologyData` for a short complex `S` that has homology -/
noncomputable def homologyData [HasHomology S] :
  S.HomologyData := HasHomology.condition.some


lemma HasHomology.mk' (h : S.HomologyData) : HasHomology S :=
  ⟨Nonempty.intro h⟩


instance [HasHomology S] : HasHomology S.op :=
  HasHomology.mk' S.homologyData.op


instance (S : ShortComplex Cᵒᵖ) [HasHomology S] : HasHomology S.unop :=
  HasHomology.mk' S.homologyData.unop


instance hasLeftHomology_of_hasHomology [S.HasHomology] : S.HasLeftHomology :=
  HasLeftHomology.mk' S.homologyData.left


instance hasRightHomology_of_hasHomology [S.HasHomology] : S.HasRightHomology :=
  HasRightHomology.mk' S.homologyData.right


instance hasHomology_of_hasCokernel {X Y : C} (f : X ⟶ Y) (Z : C) [HasCokernel f] :
    (ShortComplex.mk f (0 : Y ⟶ Z) comp_zero).HasHomology :=
  HasHomology.mk' (HomologyData.ofHasCokernel _ rfl)


instance hasHomology_of_hasKernel {Y Z : C} (g : Y ⟶ Z) (X : C) [HasKernel g] :
    (ShortComplex.mk (0 : X ⟶ Y) g zero_comp).HasHomology :=
  HasHomology.mk' (HomologyData.ofHasKernel _ rfl)


instance hasHomology_of_zeros (X Y Z : C) :
    (ShortComplex.mk (0 : X ⟶ Y) (0 : Y ⟶ Z) zero_comp).HasHomology :=
  HasHomology.mk' (HomologyData.ofZeros _ rfl rfl)


lemma hasHomology_of_epi_of_isIso_of_mono (φ : S₁ ⟶ S₂) [HasHomology S₁]
    [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] : HasHomology S₂ :=
  HasHomology.mk' (HomologyData.ofEpiOfIsIsoOfMono φ S₁.homologyData)


lemma hasHomology_of_epi_of_isIso_of_mono' (φ : S₁ ⟶ S₂) [HasHomology S₂]
    [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] : HasHomology S₁ :=
  HasHomology.mk' (HomologyData.ofEpiOfIsIsoOfMono' φ S₂.homologyData)


lemma hasHomology_of_iso (e : S₁ ≅ S₂) [HasHomology S₁] : HasHomology S₂ :=
  HasHomology.mk' (HomologyData.ofIso e S₁.homologyData)


/-- The homology map data associated to the identity morphism of a short complex. -/
@[simps]
def id (h : S.HomologyData) : HomologyMapData (𝟙 S) h h where
  left := LeftHomologyMapData.id h.left
  right := RightHomologyMapData.id h.right


/-- The homology map data associated to the zero morphism between two short complexes. -/
@[simps]
def zero (h₁ : S₁.HomologyData) (h₂ : S₂.HomologyData) :
    HomologyMapData 0 h₁ h₂ where
  left := LeftHomologyMapData.zero h₁.left h₂.left
  right := RightHomologyMapData.zero h₁.right h₂.right


/-- The composition of homology map data. -/
@[simps]
def comp {φ : S₁ ⟶ S₂} {φ' : S₂ ⟶ S₃} {h₁ : S₁.HomologyData}
    {h₂ : S₂.HomologyData} {h₃ : S₃.HomologyData}
    (ψ : HomologyMapData φ h₁ h₂) (ψ' : HomologyMapData φ' h₂ h₃) :
    HomologyMapData (φ ≫ φ') h₁ h₃ where
  left := ψ.left.comp ψ'.left
  right := ψ.right.comp ψ'.right


/-- A homology map data for a morphism of short complexes induces
a homology map data in the opposite category. -/
@[simps]
def op {φ : S₁ ⟶ S₂} {h₁ : S₁.HomologyData} {h₂ : S₂.HomologyData}
    (ψ : HomologyMapData φ h₁ h₂) :
    HomologyMapData (opMap φ) h₂.op h₁.op where
  left := ψ.right.op
  right := ψ.left.op


/-- A homology map data for a morphism of short complexes in the opposite category
induces a homology map data in the original category. -/
@[simps]
def unop {S₁ S₂ : ShortComplex Cᵒᵖ} {φ : S₁ ⟶ S₂}
    {h₁ : S₁.HomologyData} {h₂ : S₂.HomologyData}
    (ψ : HomologyMapData φ h₁ h₂) :
    HomologyMapData (unopMap φ) h₂.unop h₁.unop where
  left := ψ.right.unop
  right := ψ.left.unop


/-- When `S₁.f`, `S₁.g`, `S₂.f` and `S₂.g` are all zero, the action on homology of a
morphism `φ : S₁ ⟶ S₂` is given by the action `φ.τ₂` on the middle objects. -/
@[simps]
def ofZeros (φ : S₁ ⟶ S₂) (hf₁ : S₁.f = 0) (hg₁ : S₁.g = 0) (hf₂ : S₂.f = 0) (hg₂ : S₂.g = 0) :
    HomologyMapData φ (HomologyData.ofZeros S₁ hf₁ hg₁) (HomologyData.ofZeros S₂ hf₂ hg₂) where
  left := LeftHomologyMapData.ofZeros φ hf₁ hg₁ hf₂ hg₂
  right := RightHomologyMapData.ofZeros φ hf₁ hg₁ hf₂ hg₂


/-- When `S₁.g` and `S₂.g` are zero and we have chosen colimit cokernel coforks `c₁` and `c₂`
for `S₁.f` and `S₂.f` respectively, the action on homology of a morphism `φ : S₁ ⟶ S₂` of
short complexes is given by the unique morphism `f : c₁.pt ⟶ c₂.pt` such that
`φ.τ₂ ≫ c₂.π = c₁.π ≫ f`. -/
@[simps]
def ofIsColimitCokernelCofork (φ : S₁ ⟶ S₂)
    (hg₁ : S₁.g = 0) (c₁ : CokernelCofork S₁.f) (hc₁ : IsColimit c₁)
    (hg₂ : S₂.g = 0) (c₂ : CokernelCofork S₂.f) (hc₂ : IsColimit c₂) (f : c₁.pt ⟶ c₂.pt)
    (comm : φ.τ₂ ≫ c₂.π = c₁.π ≫ f) :
    HomologyMapData φ (HomologyData.ofIsColimitCokernelCofork S₁ hg₁ c₁ hc₁)
      (HomologyData.ofIsColimitCokernelCofork S₂ hg₂ c₂ hc₂) where
  left := LeftHomologyMapData.ofIsColimitCokernelCofork φ hg₁ c₁ hc₁ hg₂ c₂ hc₂ f comm
  right := RightHomologyMapData.ofIsColimitCokernelCofork φ hg₁ c₁ hc₁ hg₂ c₂ hc₂ f comm


/-- When `S₁.f` and `S₂.f` are zero and we have chosen limit kernel forks `c₁` and `c₂`
for `S₁.g` and `S₂.g` respectively, the action on homology of a morphism `φ : S₁ ⟶ S₂` of
short complexes is given by the unique morphism `f : c₁.pt ⟶ c₂.pt` such that
`c₁.ι ≫ φ.τ₂ = f ≫ c₂.ι`. -/
@[simps]
def ofIsLimitKernelFork (φ : S₁ ⟶ S₂)
    (hf₁ : S₁.f = 0) (c₁ : KernelFork S₁.g) (hc₁ : IsLimit c₁)
    (hf₂ : S₂.f = 0) (c₂ : KernelFork S₂.g) (hc₂ : IsLimit c₂) (f : c₁.pt ⟶ c₂.pt)
    (comm : c₁.ι ≫ φ.τ₂ = f ≫ c₂.ι) :
    HomologyMapData φ (HomologyData.ofIsLimitKernelFork S₁ hf₁ c₁ hc₁)
      (HomologyData.ofIsLimitKernelFork S₂ hf₂ c₂ hc₂) where
  left := LeftHomologyMapData.ofIsLimitKernelFork φ hf₁ c₁ hc₁ hf₂ c₂ hc₂ f comm
  right := RightHomologyMapData.ofIsLimitKernelFork φ hf₁ c₁ hc₁ hf₂ c₂ hc₂ f comm


/-- When both maps `S.f` and `S.g` of a short complex `S` are zero, this is the homology map
data (for the identity of `S`) which relates the homology data `ofZeros` and
`ofIsColimitCokernelCofork`. -/
def compatibilityOfZerosOfIsColimitCokernelCofork (hf : S.f = 0) (hg : S.g = 0)
    (c : CokernelCofork S.f) (hc : IsColimit c) :
    HomologyMapData (𝟙 S) (HomologyData.ofZeros S hf hg)
      (HomologyData.ofIsColimitCokernelCofork S hg c hc) where
  left := LeftHomologyMapData.compatibilityOfZerosOfIsColimitCokernelCofork S hf hg c hc
  right := RightHomologyMapData.compatibilityOfZerosOfIsColimitCokernelCofork S hf hg c hc


/-- When both maps `S.f` and `S.g` of a short complex `S` are zero, this is the homology map
data (for the identity of `S`) which relates the homology data
`HomologyData.ofIsLimitKernelFork` and `ofZeros` . -/
@[simps]
def compatibilityOfZerosOfIsLimitKernelFork (hf : S.f = 0) (hg : S.g = 0)
    (c : KernelFork S.g) (hc : IsLimit c) :
    HomologyMapData (𝟙 S)
      (HomologyData.ofIsLimitKernelFork S hf c hc)
      (HomologyData.ofZeros S hf hg) where
  left := LeftHomologyMapData.compatibilityOfZerosOfIsLimitKernelFork S hf hg c hc
  right := RightHomologyMapData.compatibilityOfZerosOfIsLimitKernelFork S hf hg c hc


/-- This homology map data expresses compatibilities of the homology data
constructed by `HomologyData.ofEpiOfIsIsoOfMono` -/
noncomputable def ofEpiOfIsIsoOfMono (φ : S₁ ⟶ S₂) (h : HomologyData S₁)
    [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] :
    HomologyMapData φ h (HomologyData.ofEpiOfIsIsoOfMono φ h) where
  left := LeftHomologyMapData.ofEpiOfIsIsoOfMono φ h.left
  right := RightHomologyMapData.ofEpiOfIsIsoOfMono φ h.right


/-- This homology map data expresses compatibilities of the homology data
constructed by `HomologyData.ofEpiOfIsIsoOfMono'` -/
noncomputable def ofEpiOfIsIsoOfMono' (φ : S₁ ⟶ S₂) (h : HomologyData S₂)
    [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] :
    HomologyMapData φ (HomologyData.ofEpiOfIsIsoOfMono' φ h) h where
  left := LeftHomologyMapData.ofEpiOfIsIsoOfMono' φ h.left
  right := RightHomologyMapData.ofEpiOfIsIsoOfMono' φ h.right


/-- The homology of a short complex is the `left.H` field of a chosen homology data. -/
noncomputable def homology [HasHomology S] : C := S.homologyData.left.H


/-- When a short complex has homology, this is the canonical isomorphism
`S.leftHomology ≅ S.homology`. -/
noncomputable def leftHomologyIso [S.HasHomology] : S.leftHomology ≅ S.homology :=
  leftHomologyMapIso' (Iso.refl _) _ _


/-- When a short complex has homology, this is the canonical isomorphism
`S.rightHomology ≅ S.homology`. -/
noncomputable def rightHomologyIso [S.HasHomology] : S.rightHomology ≅ S.homology :=
  rightHomologyMapIso' (Iso.refl _) _ _ ≪≫ S.homologyData.iso.symm


/-- When a short complex has homology, its homology can be computed using
any left homology data. -/
noncomputable def LeftHomologyData.homologyIso (h : S.LeftHomologyData) [S.HasHomology] :
    S.homology ≅ h.H := S.leftHomologyIso.symm ≪≫ h.leftHomologyIso


/-- When a short complex has homology, its homology can be computed using
any right homology data. -/
noncomputable def RightHomologyData.homologyIso (h : S.RightHomologyData) [S.HasHomology] :
    S.homology ≅ h.H := S.rightHomologyIso.symm ≪≫ h.rightHomologyIso


@[simp]
lemma LeftHomologyData.homologyIso_leftHomologyData [S.HasHomology] :
    S.leftHomologyData.homologyIso = S.leftHomologyIso.symm := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Eq S.leftHomologyData.homologyIso S.leftHomologyIso.symm
  -/
  ext
  /-
    case w
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Eq S.leftHomologyData.homologyIso.hom S.leftHomologyIso.symm.hom
  -/
  dsimp [homologyIso, leftHomologyIso, ShortComplex.leftHomologyIso]
  /-
    case w
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.leftHomo …
  -/
  rw [← leftHomologyMap'_comp, comp_id]
  /-
    🎉 no goals
  -/


@[simp]
lemma RightHomologyData.homologyIso_rightHomologyData [S.HasHomology] :
    S.rightHomologyData.homologyIso = S.rightHomologyIso.symm := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Eq S.rightHomologyData.homologyIso S.rightHomologyIso.symm
  -/
  ext
  /-
    case w
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Eq S.rightHomologyData.homologyIso.hom S.rightHomologyIso.symm.hom
  -/
  dsimp [homologyIso, rightHomologyIso]
  /-
    case w
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.rightHomologyIso.inv (CategoryTheor …
  -/
  erw [rightHomologyMap'_id, comp_id]
  /-
    🎉 no goals
  -/


/-- Given a morphism `φ : S₁ ⟶ S₂` of short complexes and homology data `h₁` and `h₂`
for `S₁` and `S₂` respectively, this is the induced homology map `h₁.left.H ⟶ h₁.left.H`. -/
def homologyMap' (φ : S₁ ⟶ S₂) (h₁ : S₁.HomologyData) (h₂ : S₂.HomologyData) :
    h₁.left.H ⟶ h₂.left.H := leftHomologyMap' φ _ _


/-- The homology map `S₁.homology ⟶ S₂.homology` induced by a morphism
`S₁ ⟶ S₂` of short complexes. -/
noncomputable def homologyMap (φ : S₁ ⟶ S₂) [HasHomology S₁] [HasHomology S₂] :
    S₁.homology ⟶ S₂.homology :=
  homologyMap' φ _ _


lemma homologyMap'_eq : homologyMap' φ h₁ h₂ = γ.left.φH :=
  LeftHomologyMapData.congr_φH (Subsingleton.elim _ _)


lemma cyclesMap'_eq : cyclesMap' φ h₁.left h₂.left = γ.left.φK :=
  LeftHomologyMapData.congr_φK (Subsingleton.elim _ _)


lemma opcyclesMap'_eq : opcyclesMap' φ h₁.right h₂.right = γ.right.φQ :=
  RightHomologyMapData.congr_φQ (Subsingleton.elim _ _)


lemma homologyMap_eq :
    homologyMap φ = h₁.homologyIso.hom ≫ γ.φH ≫ h₂.homologyIso.inv := by
  dsimp [homologyMap, LeftHomologyData.homologyIso, leftHomologyIso,
    LeftHomologyData.leftHomologyIso, homologyMap']
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    ⊢ Eq (CategoryTheory.ShortComplex.leftHomologyMap' φ S₁.homologyData.left S₂.h …
  -/
  simp only [← γ.leftHomologyMap'_eq, ← leftHomologyMap'_comp, id_comp, comp_id]
  /-
    🎉 no goals
  -/


lemma homologyMap_comm :
    homologyMap φ ≫ h₂.homologyIso.hom = h₁.homologyIso.hom ≫ γ.φH := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.homology …
  -/
  simp only [γ.homologyMap_eq, assoc, Iso.inv_hom_id, comp_id]
  /-
    🎉 no goals
  -/


lemma homologyMap_eq :
    homologyMap φ = h₁.homologyIso.hom ≫ γ.φH ≫ h₂.homologyIso.inv := by
  dsimp [homologyMap, homologyMap', RightHomologyData.homologyIso,
    rightHomologyIso, RightHomologyData.rightHomologyIso]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    ⊢ Eq (CategoryTheory.ShortComplex.leftHomologyMap' φ S₁.homologyData.left S₂.h …
  -/
  have γ' : HomologyMapData φ S₁.homologyData S₂.homologyData := default
  simp only [← γ.rightHomologyMap'_eq, assoc, ← rightHomologyMap'_comp_assoc,
    id_comp, comp_id, γ'.left.leftHomologyMap'_eq, γ'.right.rightHomologyMap'_eq, ← γ'.comm_assoc,
    Iso.hom_inv_id]


lemma homologyMap_comm :
    homologyMap φ ≫ h₂.homologyIso.hom = h₁.homologyIso.hom ≫ γ.φH := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.homology …
  -/
  simp only [γ.homologyMap_eq, assoc, Iso.inv_hom_id, comp_id]
  /-
    🎉 no goals
  -/


@[simp]
lemma homologyMap'_id (h : S.HomologyData) :
    homologyMap' (𝟙 S) h h = 𝟙 _ :=
  (HomologyMapData.id h).homologyMap'_eq


@[simp]
lemma homologyMap_id [HasHomology S] :
    homologyMap (𝟙 S) = 𝟙 _ :=
  homologyMap'_id _


@[simp]
lemma homologyMap'_zero (h₁ : S₁.HomologyData) (h₂ : S₂.HomologyData) :
    homologyMap' 0 h₁ h₂ = 0 :=
  (HomologyMapData.zero h₁ h₂).homologyMap'_eq


@[simp]
lemma homologyMap_zero [S₁.HasHomology] [S₂.HasHomology] :
    homologyMap (0 : S₁ ⟶ S₂) = 0 :=
  homologyMap'_zero _ _


lemma homologyMap'_comp (φ₁ : S₁ ⟶ S₂) (φ₂ : S₂ ⟶ S₃)
    (h₁ : S₁.HomologyData) (h₂ : S₂.HomologyData) (h₃ : S₃.HomologyData) :
    homologyMap' (φ₁ ≫ φ₂) h₁ h₃ = homologyMap' φ₁ h₁ h₂ ≫
      homologyMap' φ₂ h₂ h₃ :=
  leftHomologyMap'_comp _ _ _ _ _


@[simp]
lemma homologyMap_comp [HasHomology S₁] [HasHomology S₂] [HasHomology S₃]
    (φ₁ : S₁ ⟶ S₂) (φ₂ : S₂ ⟶ S₃) :
    homologyMap (φ₁ ≫ φ₂) = homologyMap φ₁ ≫ homologyMap φ₂ :=
  homologyMap'_comp _ _ _ _ _


/-- Given an isomorphism `S₁ ≅ S₂` of short complexes and homology data `h₁` and `h₂`
for `S₁` and `S₂` respectively, this is the induced homology isomorphism `h₁.left.H ≅ h₁.left.H`. -/
@[simps]
def homologyMapIso' (e : S₁ ≅ S₂) (h₁ : S₁.HomologyData)
    (h₂ : S₂.HomologyData) : h₁.left.H ≅ h₂.left.H where
  hom := homologyMap' e.hom h₁ h₂
  inv := homologyMap' e.inv h₂ h₁
                   /-
                     C : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
                     φ : Quiver.Hom S₁ S₂
                     h₁✝ : S₁.HomologyData
                     h₂✝ : S₂.HomologyData
                     e : CategoryTheory.Iso S₁ S₂
                     h₁ : S₁.HomologyData
                     h₂ : S₂.HomologyData
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.homology …
                   -/
  hom_inv_id := by rw [← homologyMap'_comp, e.hom_inv_id, homologyMap'_id]
                   /-
                     🎉 no goals
                   -/
                   /-
                     C : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
                     φ : Quiver.Hom S₁ S₂
                     h₁✝ : S₁.HomologyData
                     h₂✝ : S₂.HomologyData
                     e : CategoryTheory.Iso S₁ S₂
                     h₁ : S₁.HomologyData
                     h₂ : S₂.HomologyData
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.homology …
                   -/
  inv_hom_id := by rw [← homologyMap'_comp, e.inv_hom_id, homologyMap'_id]
                   /-
                     🎉 no goals
                   -/


instance isIso_homologyMap'_of_isIso (φ : S₁ ⟶ S₂) [IsIso φ]
    (h₁ : S₁.HomologyData) (h₂ : S₂.HomologyData) :
    IsIso (homologyMap' φ h₁ h₂) :=
  (inferInstance : IsIso (homologyMapIso' (asIso φ) h₁ h₂).hom)


/-- The homology isomorphism `S₁.homology ⟶ S₂.homology` induced by an isomorphism
`S₁ ≅ S₂` of short complexes. -/
@[simps]
noncomputable def homologyMapIso (e : S₁ ≅ S₂) [S₁.HasHomology]
    [S₂.HasHomology] : S₁.homology ≅ S₂.homology where
  hom := homologyMap e.hom
  inv := homologyMap e.inv
                   /-
                     C : Type u
                     inst✝³ : CategoryTheory.Category.{v, u} C
                     inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                     S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
                     φ : Quiver.Hom S₁ S₂
                     h₁ : S₁.HomologyData
                     h₂ : S₂.HomologyData
                     e : CategoryTheory.Iso S₁ S₂
                     inst✝¹ : S₁.HasHomology
                     inst✝ : S₂.HasHomology
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.homology …
                   -/
  hom_inv_id := by rw [← homologyMap_comp, e.hom_inv_id, homologyMap_id]
                   /-
                     🎉 no goals
                   -/
                   /-
                     C : Type u
                     inst✝³ : CategoryTheory.Category.{v, u} C
                     inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                     S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
                     φ : Quiver.Hom S₁ S₂
                     h₁ : S₁.HomologyData
                     h₂ : S₂.HomologyData
                     e : CategoryTheory.Iso S₁ S₂
                     inst✝¹ : S₁.HasHomology
                     inst✝ : S₂.HasHomology
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.homology …
                   -/
  inv_hom_id := by rw [← homologyMap_comp, e.inv_hom_id, homologyMap_id]
                   /-
                     🎉 no goals
                   -/


instance isIso_homologyMap_of_iso (φ : S₁ ⟶ S₂) [IsIso φ] [S₁.HasHomology]
    [S₂.HasHomology] :
    IsIso (homologyMap φ) :=
  (inferInstance : IsIso (homologyMapIso (asIso φ)).hom)


/-- If a short complex `S` has both a left homology data `h₁` and a right homology data `h₂`,
this is the canonical morphism `h₁.H ⟶ h₂.H`. -/
def leftRightHomologyComparison' : h₁.H ⟶ h₂.H :=
                                       /-
                                         C : Type u
                                         inst✝¹ : CategoryTheory.Category.{v, u} C
                                         inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                         S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
                                         φ : Quiver.Hom S₁ S₂
                                         h₁✝ : S₁.HomologyData
                                         h₂✝ : S₂.HomologyData
                                         h₁ : S.LeftHomologyData
                                         h₂ : S.RightHomologyData
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp h₁.f' (CategoryTheory.CategoryStruct. …
                                       -/
  h₂.liftH (h₁.descH (h₁.i ≫ h₂.p) (by simp))
                                       /-
                                         🎉 no goals
                                       -/
    (by rw [← cancel_epi h₁.π, LeftHomologyData.π_descH_assoc, assoc,
      RightHomologyData.p_g', LeftHomologyData.wi, comp_zero])


lemma leftRightHomologyComparison'_eq_liftH :
    leftRightHomologyComparison' h₁ h₂ =
                                           /-
                                             C : Type u
                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                             inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                             S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
                                             φ : Quiver.Hom S₁ S₂
                                             h₁✝ : S₁.HomologyData
                                             h₂✝ : S₂.HomologyData
                                             h₁ : S.LeftHomologyData
                                             h₂ : S.RightHomologyData
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp h₁.f' (CategoryTheory.CategoryStruct. …
                                           -/
      h₂.liftH (h₁.descH (h₁.i ≫ h₂.p) (by simp))
                                           /-
                                             🎉 no goals
                                           -/
        (by rw [← cancel_epi h₁.π, LeftHomologyData.π_descH_assoc, assoc,
          RightHomologyData.p_g', LeftHomologyData.wi, comp_zero]) := rfl


@[reassoc (attr := simp)]
lemma π_leftRightHomologyComparison'_ι :
    h₁.π ≫ leftRightHomologyComparison' h₁ h₂ ≫ h₂.ι = h₁.i ≫ h₂.p := by
  simp only [leftRightHomologyComparison'_eq_liftH,
    RightHomologyData.liftH_ι, LeftHomologyData.π_descH]


lemma leftRightHomologyComparison'_eq_descH :
    leftRightHomologyComparison' h₁ h₂ =
                                           /-
                                             C : Type u
                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                             inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                             S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
                                             φ : Quiver.Hom S₁ S₂
                                             h₁✝ : S₁.HomologyData
                                             h₂✝ : S₂.HomologyData
                                             h₁ : S.LeftHomologyData
                                             h₂ : S.RightHomologyData
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp h …
                                           -/
      h₁.descH (h₂.liftH (h₁.i ≫ h₂.p) (by simp))
                                           /-
                                             🎉 no goals
                                           -/
        (by rw [← cancel_mono h₂.ι, assoc, RightHomologyData.liftH_ι,
          LeftHomologyData.f'_i_assoc, RightHomologyData.wp, zero_comp]) := by
  simp only [← cancel_mono h₂.ι, ← cancel_epi h₁.π, π_leftRightHomologyComparison'_ι,
    LeftHomologyData.π_descH_assoc, RightHomologyData.liftH_ι]


/-- If a short complex `S` has both a left and right homology,
this is the canonical morphism `S.leftHomology ⟶ S.rightHomology`. -/
noncomputable def leftRightHomologyComparison [S.HasLeftHomology] [S.HasRightHomology] :
    S.leftHomology ⟶ S.rightHomology :=
  leftRightHomologyComparison' _ _


@[reassoc (attr := simp)]
lemma π_leftRightHomologyComparison_ι [S.HasLeftHomology] [S.HasRightHomology] :
    S.leftHomologyπ ≫ S.leftRightHomologyComparison ≫ S.rightHomologyι =
      S.iCycles ≫ S.pOpcycles :=
  π_leftRightHomologyComparison'_ι _ _


@[reassoc]
lemma leftRightHomologyComparison'_naturality (φ : S₁ ⟶ S₂) (h₁ : S₁.LeftHomologyData)
    (h₂ : S₁.RightHomologyData) (h₁' : S₂.LeftHomologyData) (h₂' : S₂.RightHomologyData) :
    leftHomologyMap' φ h₁ h₁' ≫ leftRightHomologyComparison' h₁' h₂' =
      leftRightHomologyComparison' h₁ h₂ ≫ rightHomologyMap' φ h₂ h₂' := by
  simp only [← cancel_epi h₁.π, ← cancel_mono h₂'.ι, assoc,
    leftHomologyπ_naturality'_assoc, rightHomologyι_naturality',
    π_leftRightHomologyComparison'_ι, π_leftRightHomologyComparison'_ι_assoc,
    cyclesMap'_i_assoc, p_opcyclesMap']


lemma leftRightHomologyComparison'_compatibility (h₁ h₁' : S.LeftHomologyData)
    (h₂ h₂' : S.RightHomologyData) :
    leftRightHomologyComparison' h₁ h₂ = leftHomologyMap' (𝟙 S) h₁ h₁' ≫
      leftRightHomologyComparison' h₁' h₂' ≫ rightHomologyMap' (𝟙 S) _ _ := by
  rw [leftRightHomologyComparison'_naturality_assoc (𝟙 S) h₁ h₂ h₁' h₂',
    ← rightHomologyMap'_comp, comp_id, rightHomologyMap'_id, comp_id]


lemma leftRightHomologyComparison_eq [S.HasLeftHomology] [S.HasRightHomology]
    (h₁ : S.LeftHomologyData) (h₂ : S.RightHomologyData) :
    S.leftRightHomologyComparison = h₁.leftHomologyIso.hom ≫
      leftRightHomologyComparison' h₁ h₂ ≫ h₂.rightHomologyIso.inv :=
  leftRightHomologyComparison'_compatibility _ _ _ _


@[simp]
lemma HomologyData.leftRightHomologyComparison'_eq (h : S.HomologyData) :
    leftRightHomologyComparison' h.left h.right = h.iso.hom := by
  simp only [← cancel_epi h.left.π, ← cancel_mono h.right.ι, assoc,
    π_leftRightHomologyComparison'_ι, comm]


instance isIso_leftRightHomologyComparison'_of_homologyData (h : S.HomologyData) :
  IsIso (leftRightHomologyComparison' h.left h.right) := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.HomologyData
      h₂ : S₂.HomologyData
      h : S.HomologyData
      ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.leftRightHomologyCompariso …
    -/
    rw [h.leftRightHomologyComparison'_eq]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.HomologyData
      h₂ : S₂.HomologyData
      h : S.HomologyData
      ⊢ CategoryTheory.IsIso h.iso.hom
    -/
    infer_instance
    /-
      🎉 no goals
    -/


instance isIso_leftRightHomologyComparison' [S.HasHomology]
    (h₁ : S.LeftHomologyData) (h₂ : S.RightHomologyData) :
    IsIso (leftRightHomologyComparison' h₁ h₂) := by
  rw [leftRightHomologyComparison'_compatibility h₁ S.homologyData.left h₂
    S.homologyData.right]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁✝ : S₁.HomologyData
    h₂✝ : S₂.HomologyData
    inst✝ : S.HasHomology
    h₁ : S.LeftHomologyData
    h₂ : S.RightHomologyData
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Sho …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance isIso_leftRightHomologyComparison [S.HasHomology] :
    IsIso S.leftRightHomologyComparison := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.HomologyData
    h₂ : S₂.HomologyData
    inst✝ : S.HasHomology
    ⊢ CategoryTheory.IsIso S.leftRightHomologyComparison
  -/
  dsimp only [leftRightHomologyComparison]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.HomologyData
    h₂ : S₂.HomologyData
    inst✝ : S.HasHomology
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.leftRightHomologyCompariso …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- This is the homology data for a short complex `S` that is obtained
from a left homology data `h₁` and a right homology data `h₂` when the comparison
morphism `leftRightHomologyComparison' h₁ h₂ : h₁.H ⟶ h₂.H` is an isomorphism. -/
@[simps]
noncomputable def ofIsIsoLeftRightHomologyComparison'
    (h₁ : S.LeftHomologyData) (h₂ : S.RightHomologyData)
    [IsIso (leftRightHomologyComparison' h₁ h₂)] :
    S.HomologyData where
  left := h₁
  right := h₂
  iso := asIso (leftRightHomologyComparison' h₁ h₂)


lemma leftRightHomologyComparison'_eq_leftHomologpMap'_comp_iso_hom_comp_rightHomologyMap'
    (h : S.HomologyData) (h₁ : S.LeftHomologyData) (h₂ : S.RightHomologyData) :
    leftRightHomologyComparison' h₁ h₂ =
      leftHomologyMap' (𝟙 S) h₁ h.left ≫ h.iso.hom ≫ rightHomologyMap' (𝟙 S) h.right h₂ := by
  simpa only [h.leftRightHomologyComparison'_eq] using
    leftRightHomologyComparison'_compatibility h₁ h.left h₂ h.right


@[reassoc]
lemma leftRightHomologyComparison'_fac (h₁ : S.LeftHomologyData) (h₂ : S.RightHomologyData)
    [S.HasHomology] :
    leftRightHomologyComparison' h₁ h₂ = h₁.homologyIso.inv ≫ h₂.homologyIso.hom := by
  rw [leftRightHomologyComparison'_eq_leftHomologpMap'_comp_iso_hom_comp_rightHomologyMap'
    S.homologyData h₁ h₂]
  dsimp only [LeftHomologyData.homologyIso, LeftHomologyData.leftHomologyIso,
    Iso.symm, Iso.trans, Iso.refl, leftHomologyMapIso', leftHomologyIso,
    RightHomologyData.homologyIso, RightHomologyData.rightHomologyIso,
    rightHomologyMapIso', rightHomologyIso]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h₁ : S.LeftHomologyData
    h₂ : S.RightHomologyData
    inst✝ : S.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.leftHomo …
  -/
  simp only [assoc, ← leftHomologyMap'_comp_assoc, id_comp, ← rightHomologyMap'_comp]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma leftRightHomologyComparison_fac [S.HasHomology] :
    S.leftRightHomologyComparison = S.leftHomologyIso.hom ≫ S.rightHomologyIso.inv := by
  simpa only [LeftHomologyData.homologyIso_leftHomologyData, Iso.symm_inv,
    RightHomologyData.homologyIso_rightHomologyData, Iso.symm_hom] using
      leftRightHomologyComparison'_fac S.leftHomologyData S.rightHomologyData


lemma HomologyData.right_homologyIso_eq_left_homologyIso_trans_iso
    (h : S.HomologyData) [S.HasHomology] :
    h.right.homologyIso = h.left.homologyIso ≪≫ h.iso := by
  suffices h.iso = h.left.homologyIso.symm ≪≫ h.right.homologyIso by
    rw [this, Iso.self_symm_id_assoc]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.HomologyData
    inst✝ : S.HasHomology
    ⊢ Eq h.iso (h.left.homologyIso.symm.trans h.right.homologyIso)
  -/
  ext
  /-
    case w
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.HomologyData
    inst✝ : S.HasHomology
    ⊢ Eq h.iso.hom (h.left.homologyIso.symm.trans h.right.homologyIso).hom
  -/
  dsimp
  /-
    case w
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.HomologyData
    inst✝ : S.HasHomology
    ⊢ Eq h.iso.hom (CategoryTheory.CategoryStruct.comp h.left.homologyIso.inv h.ri …
  -/
  rw [← leftRightHomologyComparison'_fac, leftRightHomologyComparison'_eq]
  /-
    🎉 no goals
  -/


lemma hasHomology_of_isIso_leftRightHomologyComparison'
    (h₁ : S.LeftHomologyData) (h₂ : S.RightHomologyData)
    [IsIso (leftRightHomologyComparison' h₁ h₂)] :
    S.HasHomology :=
  HasHomology.mk' (HomologyData.ofIsIsoLeftRightHomologyComparison' h₁ h₂)


lemma hasHomology_of_isIsoLeftRightHomologyComparison [S.HasLeftHomology]
    [S.HasRightHomology] [h : IsIso S.leftRightHomologyComparison] :
    S.HasHomology := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : S.HasLeftHomology
    inst✝ : S.HasRightHomology
    h : CategoryTheory.IsIso S.leftRightHomologyComparison
    ⊢ S.HasHomology
  -/
  haveI : IsIso (leftRightHomologyComparison' S.leftHomologyData S.rightHomologyData) := h
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : S.HasLeftHomology
    inst✝ : S.HasRightHomology
    h : CategoryTheory.IsIso S.leftRightHomologyComparison
    this : CategoryTheory.IsIso (CategoryTheory.ShortComplex.leftRightHomologyComp …
    ⊢ S.HasHomology
  -/
  exact hasHomology_of_isIso_leftRightHomologyComparison' S.leftHomologyData S.rightHomologyData
  /-
    🎉 no goals
  -/


@[reassoc]
lemma LeftHomologyData.leftHomologyIso_hom_naturality
    (h₁ : S₁.LeftHomologyData) (h₂ : S₂.LeftHomologyData) :
    h₁.homologyIso.hom ≫ leftHomologyMap' φ h₁ h₂ =
      homologyMap φ ≫ h₂.homologyIso.hom := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h₁.homologyIso.hom (CategoryTheory.Sh …
  -/
  dsimp [homologyIso, ShortComplex.leftHomologyIso, homologyMap, homologyMap', leftHomologyIso]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [← leftHomologyMap'_comp, id_comp, comp_id]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma LeftHomologyData.leftHomologyIso_inv_naturality
    (h₁ : S₁.LeftHomologyData) (h₂ : S₂.LeftHomologyData) :
    h₁.homologyIso.inv ≫ homologyMap φ =
      leftHomologyMap' φ h₁ h₂ ≫ h₂.homologyIso.inv := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h₁.homologyIso.inv (CategoryTheory.Sh …
  -/
  dsimp [homologyIso, ShortComplex.leftHomologyIso, homologyMap, homologyMap', leftHomologyIso]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [← leftHomologyMap'_comp, id_comp, comp_id]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma leftHomologyIso_hom_naturality :
    S₁.leftHomologyIso.hom ≫ homologyMap φ =
      leftHomologyMap φ ≫ S₂.leftHomologyIso.hom := by
  simpa only [LeftHomologyData.homologyIso_leftHomologyData, Iso.symm_inv] using
    LeftHomologyData.leftHomologyIso_inv_naturality φ S₁.leftHomologyData S₂.leftHomologyData


@[reassoc]
lemma leftHomologyIso_inv_naturality :
    S₁.leftHomologyIso.inv ≫ leftHomologyMap φ =
      homologyMap φ ≫ S₂.leftHomologyIso.inv := by
  simpa only [LeftHomologyData.homologyIso_leftHomologyData, Iso.symm_inv] using
    LeftHomologyData.leftHomologyIso_hom_naturality φ S₁.leftHomologyData S₂.leftHomologyData


@[reassoc]
lemma RightHomologyData.rightHomologyIso_hom_naturality
    (h₁ : S₁.RightHomologyData) (h₂ : S₂.RightHomologyData) :
    h₁.homologyIso.hom ≫ rightHomologyMap' φ h₁ h₂ =
      homologyMap φ ≫ h₂.homologyIso.hom := by
  rw [← cancel_epi h₁.homologyIso.inv, Iso.inv_hom_id_assoc,
    ← cancel_epi (leftRightHomologyComparison' S₁.leftHomologyData h₁),
    ← leftRightHomologyComparison'_naturality φ S₁.leftHomologyData h₁ S₂.leftHomologyData h₂,
    ← cancel_epi (S₁.leftHomologyData.homologyIso.hom),
    LeftHomologyData.leftHomologyIso_hom_naturality_assoc,
    leftRightHomologyComparison'_fac, leftRightHomologyComparison'_fac, assoc,
    Iso.hom_inv_id_assoc, Iso.hom_inv_id_assoc, Iso.hom_inv_id_assoc]


@[reassoc]
lemma RightHomologyData.rightHomologyIso_inv_naturality
    (h₁ : S₁.RightHomologyData) (h₂ : S₂.RightHomologyData) :
      h₁.homologyIso.inv ≫ homologyMap φ =
        rightHomologyMap' φ h₁ h₂ ≫ h₂.homologyIso.inv := by
  simp only [← cancel_mono h₂.homologyIso.hom, assoc, Iso.inv_hom_id_assoc, comp_id,
    ← RightHomologyData.rightHomologyIso_hom_naturality φ h₁ h₂, Iso.inv_hom_id]


@[reassoc]
lemma rightHomologyIso_hom_naturality :
    S₁.rightHomologyIso.hom ≫ homologyMap φ =
      rightHomologyMap φ ≫ S₂.rightHomologyIso.hom := by
  simpa only [RightHomologyData.homologyIso_rightHomologyData, Iso.symm_inv] using
    RightHomologyData.rightHomologyIso_inv_naturality φ S₁.rightHomologyData S₂.rightHomologyData


@[reassoc]
lemma rightHomologyIso_inv_naturality :
    S₁.rightHomologyIso.inv ≫ rightHomologyMap φ =
      homologyMap φ ≫ S₂.rightHomologyIso.inv := by
  simpa only [RightHomologyData.homologyIso_rightHomologyData, Iso.symm_inv] using
    RightHomologyData.rightHomologyIso_hom_naturality φ S₁.rightHomologyData S₂.rightHomologyData


/-- We shall say that a category `C` is a category with homology when all short complexes
have homology. -/
class _root_.CategoryTheory.CategoryWithHomology : Prop where
  hasHomology : ∀ (S : ShortComplex C), S.HasHomology


instance [CategoryWithHomology C] : CategoryWithHomology Cᵒᵖ :=
  ⟨fun S => HasHomology.mk' S.unop.homologyData.op⟩


/-- The homology functor `ShortComplex C ⥤ C` for a category `C` with homology. -/
@[simps]
noncomputable def homologyFunctor [CategoryWithHomology C] :
    ShortComplex C ⥤ C where
  obj S := S.homology
  map f := homologyMap f


instance isIso_homologyMap'_of_epi_of_isIso_of_mono (φ : S₁ ⟶ S₂)
    (h₁ : S₁.HomologyData) (h₂ : S₂.HomologyData) [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] :
    IsIso (homologyMap' φ h₁ h₂) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    φ✝ : Quiver.Hom S₁ S₂
    h₁✝ : S₁.HomologyData
    h₂✝ : S₂.HomologyData
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.HomologyData
    h₂ : S₂.HomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap' φ h₁ h₂)
  -/
  dsimp only [homologyMap']
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    φ✝ : Quiver.Hom S₁ S₂
    h₁✝ : S₁.HomologyData
    h₂✝ : S₂.HomologyData
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.HomologyData
    h₂ : S₂.HomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.leftHomologyMap' φ h₁.left …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma isIso_homologyMap_of_epi_of_isIso_of_mono' (φ : S₁ ⟶ S₂) [S₁.HasHomology] [S₂.HasHomology]
    (h₁ : Epi φ.τ₁) (h₂ : IsIso φ.τ₂) (h₃ : Mono φ.τ₃) :
    IsIso (homologyMap φ) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    h₁ : CategoryTheory.Epi φ.τ₁
    h₂ : CategoryTheory.IsIso φ.τ₂
    h₃ : CategoryTheory.Mono φ.τ₃
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ)
  -/
  dsimp only [homologyMap]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    h₁ : CategoryTheory.Epi φ.τ₁
    h₂ : CategoryTheory.IsIso φ.τ₂
    h₃ : CategoryTheory.Mono φ.τ₃
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap' φ S₁.homology …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance isIso_homologyMap_of_epi_of_isIso_of_mono (φ : S₁ ⟶ S₂) [S₁.HasHomology] [S₂.HasHomology]
    [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] :
    IsIso (homologyMap φ) :=
  isIso_homologyMap_of_epi_of_isIso_of_mono' φ inferInstance inferInstance inferInstance


instance isIso_homologyFunctor_map_of_epi_of_isIso_of_mono (φ : S₁ ⟶ S₂) [CategoryWithHomology C]
    [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] :
    IsIso ((homologyFunctor C).map φ) :=
  (inferInstance : IsIso (homologyMap φ))


instance isIso_homologyMap_of_isIso (φ : S₁ ⟶ S₂) [S₁.HasHomology] [S₂.HasHomology] [IsIso φ] :
    IsIso (homologyMap φ) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    φ✝ : Quiver.Hom S₁ S₂
    h₁ : S₁.HomologyData
    h₂ : S₂.HomologyData
    φ : Quiver.Hom S₁ S₂
    inst✝² : S₁.HasHomology
    inst✝¹ : S₂.HasHomology
    inst✝ : CategoryTheory.IsIso φ
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ)
  -/
  dsimp only [homologyMap, homologyMap']
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    φ✝ : Quiver.Hom S₁ S₂
    h₁ : S₁.HomologyData
    h₂ : S₂.HomologyData
    φ : Quiver.Hom S₁ S₂
    inst✝² : S₁.HasHomology
    inst✝¹ : S₂.HasHomology
    inst✝ : CategoryTheory.IsIso φ
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.leftHomologyMap' φ S₁.homo …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The canonical morphism `S.cycles ⟶ S.homology` for a short complex `S` that has homology. -/
noncomputable def homologyπ : S.cycles ⟶ S.homology :=
  S.leftHomologyπ ≫ S.leftHomologyIso.hom


/-- The canonical morphism `S.homology ⟶ S.opcycles` for a short complex `S` that has homology. -/
noncomputable def homologyι : S.homology ⟶ S.opcycles :=
  S.rightHomologyIso.inv ≫ S.rightHomologyι


@[reassoc (attr := simp)]
lemma homologyπ_comp_leftHomologyIso_inv :
    S.homologyπ ≫ S.leftHomologyIso.inv = S.leftHomologyπ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.homologyπ S.leftHomologyIso.inv) S. …
  -/
  dsimp only [homologyπ]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp S …
  -/
  simp only [assoc, Iso.hom_inv_id, comp_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma rightHomologyIso_hom_comp_homologyι :
    S.rightHomologyIso.hom ≫ S.homologyι = S.rightHomologyι := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.rightHomologyIso.hom S.homologyι) S …
  -/
  dsimp only [homologyι]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.rightHomologyIso.hom (CategoryTheor …
  -/
  simp only [Iso.hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma toCycles_comp_homologyπ :
    S.toCycles ≫ S.homologyπ = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.toCycles S.homologyπ) 0
  -/
  dsimp only [homologyπ]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.toCycles (CategoryTheory.CategorySt …
  -/
  simp only [toCycles_comp_leftHomologyπ_assoc, zero_comp]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma homologyι_comp_fromOpcycles :
    S.homologyι ≫ S.fromOpcycles = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.homologyι S.fromOpcycles) 0
  -/
  dsimp only [homologyι]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp S …
  -/
  simp only [assoc, rightHomologyι_comp_fromOpcycles, comp_zero]
  /-
    🎉 no goals
  -/


/-- The homology `S.homology` of a short complex is
the cokernel of the morphism `S.toCycles : S.X₁ ⟶ S.cycles`. -/
noncomputable def homologyIsCokernel :
    IsColimit (CokernelCofork.ofπ S.homologyπ S.toCycles_comp_homologyπ) :=
  IsColimit.ofIsoColimit S.leftHomologyIsCokernel
    (Cofork.ext S.leftHomologyIso rfl)


/-- The homology `S.homology` of a short complex is
the kernel of the morphism `S.fromOpcycles : S.opcycles ⟶ S.X₃`. -/
noncomputable def homologyIsKernel :
    IsLimit (KernelFork.ofι S.homologyι S.homologyι_comp_fromOpcycles) :=
  IsLimit.ofIsoLimit S.rightHomologyIsKernel
                                     /-
                                       C : Type u
                                       inst✝² : CategoryTheory.Category.{v, u} C
                                       inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                       S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
                                       φ : Quiver.Hom S₁ S₂
                                       h₁ : S₁.HomologyData
                                       h₂ : S₂.HomologyData
                                       A : C
                                       inst✝ : S.HasHomology
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp S.rightHomologyIso.hom (CategoryTheor …
                                     -/
    (Fork.ext S.rightHomologyIso (by simp))
                                     /-
                                       🎉 no goals
                                     -/


instance : Epi S.homologyπ :=
  Limits.epi_of_isColimit_cofork (S.homologyIsCokernel)


instance : Mono S.homologyι :=
  Limits.mono_of_isLimit_fork (S.homologyIsKernel)


/-- Given a morphism `k : S.cycles ⟶ A` such that `S.toCycles ≫ k = 0`, this is the
induced morphism `S.homology ⟶ A`. -/
noncomputable def descHomology (k : S.cycles ⟶ A) (hk : S.toCycles ≫ k = 0) :
    S.homology ⟶ A :=
  S.homologyIsCokernel.desc (CokernelCofork.ofπ k hk)


/-- Given a morphism `k : A ⟶ S.opcycles` such that `k ≫ S.fromOpcycles = 0`, this is the
induced morphism `A ⟶ S.homology`. -/
noncomputable def liftHomology (k : A ⟶ S.opcycles) (hk : k ≫ S.fromOpcycles = 0) :
    A ⟶ S.homology :=
  S.homologyIsKernel.lift (KernelFork.ofι k hk)


@[reassoc (attr := simp)]
lemma π_descHomology (k : S.cycles ⟶ A) (hk : S.toCycles ≫ k = 0) :
    S.homologyπ ≫ S.descHomology k hk = k :=
  Cofork.IsColimit.π_desc S.homologyIsCokernel


@[reassoc (attr := simp)]
lemma liftHomology_ι (k : A ⟶ S.opcycles) (hk : k ≫ S.fromOpcycles = 0) :
    S.liftHomology k hk ≫ S.homologyι = k :=
  Fork.IsLimit.lift_ι S.homologyIsKernel


@[reassoc (attr := simp)]
lemma homologyπ_naturality (φ : S₁ ⟶ S₂) [S₁.HasHomology] [S₂.HasHomology] :
    S₁.homologyπ ≫ homologyMap φ = cyclesMap φ ≫ S₂.homologyπ := by
  simp only [← cancel_mono S₂.leftHomologyIso.inv, assoc, ← leftHomologyIso_inv_naturality φ,
    homologyπ_comp_leftHomologyIso_inv]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S₁.homologyπ (CategoryTheory.Category …
  -/
  simp only [homologyπ, assoc, Iso.hom_inv_id_assoc, leftHomologyπ_naturality]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma homologyι_naturality (φ : S₁ ⟶ S₂) [S₁.HasHomology] [S₂.HasHomology] :
    homologyMap φ ≫ S₂.homologyι = S₁.homologyι ≫ S₁.opcyclesMap φ  := by
  simp only [← cancel_epi S₁.rightHomologyIso.hom, rightHomologyIso_hom_naturality_assoc φ,
    rightHomologyIso_hom_comp_homologyι, rightHomologyι_naturality]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S₁.rightHomologyι (CategoryTheory.Sho …
  -/
  simp only [homologyι, assoc, Iso.hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma homology_π_ι :
    S.homologyπ ≫ S.homologyι = S.iCycles ≫ S.pOpcycles := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.homologyπ S.homologyι) (CategoryThe …
  -/
  dsimp only [homologyπ, homologyι]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp S …
  -/
  simpa only [assoc, S.leftRightHomologyComparison_fac] using S.π_leftRightHomologyComparison_ι
  /-
    🎉 no goals
  -/


/-- The homology of a short complex `S` identifies to the kernel of the induced morphism
`cokernel S.f ⟶ S.X₃`. -/
noncomputable def homologyIsoKernelDesc [HasCokernel S.f]
    [HasKernel (cokernel.desc S.f S.g S.zero)] :
  S.homology ≅ kernel (cokernel.desc S.f S.g S.zero) :=
    S.rightHomologyIso.symm ≪≫ S.rightHomologyIsoKernelDesc


/-- The homology of a short complex `S` identifies to the cokernel of the induced morphism
`S.X₁ ⟶ kernel S.g`. -/
noncomputable def homologyIsoCokernelLift [HasKernel S.g]
    [HasCokernel (kernel.lift S.g S.f S.zero)] :
  S.homology ≅ cokernel (kernel.lift S.g S.f S.zero) :=
    S.leftHomologyIso.symm ≪≫ S.leftHomologyIsoCokernelLift


@[reassoc (attr := simp)]
lemma LeftHomologyData.homologyπ_comp_homologyIso_hom (h : S.LeftHomologyData) :
    S.homologyπ ≫ h.homologyIso.hom = h.cyclesIso.hom ≫ h.π := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.LeftHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.homologyπ h.homologyIso.hom) (Categ …
  -/
  dsimp only [homologyπ, homologyIso]
  simp only [Iso.trans_hom, Iso.symm_hom, assoc, Iso.hom_inv_id_assoc,
    leftHomologyπ_comp_leftHomologyIso_hom]


@[reassoc (attr := simp)]
lemma LeftHomologyData.π_comp_homologyIso_inv (h : S.LeftHomologyData) :
    h.π ≫ h.homologyIso.inv = h.cyclesIso.inv ≫ S.homologyπ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.LeftHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.π h.homologyIso.inv) (CategoryTheor …
  -/
  dsimp only [homologyπ, homologyIso]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.LeftHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.π (S.leftHomologyIso.symm.trans h.l …
  -/
  simp only [Iso.trans_inv, Iso.symm_inv, π_comp_leftHomologyIso_inv_assoc]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma RightHomologyData.homologyIso_inv_comp_homologyι (h : S.RightHomologyData) :
    h.homologyIso.inv ≫ S.homologyι = h.ι ≫ h.opcyclesIso.inv := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.RightHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.homologyIso.inv S.homologyι) (Categ …
  -/
  dsimp only [homologyι, homologyIso]
  simp only [Iso.trans_inv, Iso.symm_inv, assoc, Iso.hom_inv_id_assoc,
    rightHomologyIso_inv_comp_rightHomologyι]


@[reassoc (attr := simp)]
lemma RightHomologyData.homologyIso_hom_comp_ι (h : S.RightHomologyData) :
    h.homologyIso.hom ≫ h.ι = S.homologyι ≫ h.opcyclesIso.hom := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.RightHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.homologyIso.hom h.ι) (CategoryTheor …
  -/
  dsimp only [homologyι, homologyIso]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.RightHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.rightHomologyIso.symm.trans h.righ …
  -/
  simp only [Iso.trans_hom, Iso.symm_hom, assoc, rightHomologyIso_hom_comp_ι]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma LeftHomologyData.homologyIso_hom_comp_leftHomologyIso_inv (h : S.LeftHomologyData) :
    h.homologyIso.hom ≫ h.leftHomologyIso.inv = S.leftHomologyIso.inv := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.LeftHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.homologyIso.hom h.leftHomologyIso.i …
  -/
  dsimp only [homologyIso]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.LeftHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.leftHomologyIso.symm.trans h.leftH …
  -/
  simp only [Iso.trans_hom, Iso.symm_hom, assoc, Iso.hom_inv_id, comp_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma LeftHomologyData.leftHomologyIso_hom_comp_homologyIso_inv (h : S.LeftHomologyData) :
    h.leftHomologyIso.hom ≫ h.homologyIso.inv = S.leftHomologyIso.hom := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.LeftHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.leftHomologyIso.hom h.homologyIso.i …
  -/
  dsimp only [homologyIso]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.LeftHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.leftHomologyIso.hom (S.leftHomology …
  -/
  simp only [Iso.trans_inv, Iso.symm_inv, Iso.hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma RightHomologyData.homologyIso_hom_comp_rightHomologyIso_inv (h : S.RightHomologyData) :
    h.homologyIso.hom ≫ h.rightHomologyIso.inv = S.rightHomologyIso.inv := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.RightHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.homologyIso.hom h.rightHomologyIso. …
  -/
  dsimp only [homologyIso]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.RightHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.rightHomologyIso.symm.trans h.righ …
  -/
  simp only [Iso.trans_hom, Iso.symm_hom, assoc, Iso.hom_inv_id, comp_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma RightHomologyData.rightHomologyIso_hom_comp_homologyIso_inv (h : S.RightHomologyData) :
    h.rightHomologyIso.hom ≫ h.homologyIso.inv = S.rightHomologyIso.hom := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.RightHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.rightHomologyIso.hom h.homologyIso. …
  -/
  dsimp only [homologyIso]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.RightHomologyData
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.rightHomologyIso.hom (S.rightHomolo …
  -/
  simp only [Iso.trans_inv, Iso.symm_inv, Iso.hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma comp_homologyMap_comp [S₁.HasHomology] [S₂.HasHomology] (φ : S₁ ⟶ S₂)
    (h₁ : S₁.LeftHomologyData) (h₂ : S₂.RightHomologyData) :
    h₁.π ≫ h₁.homologyIso.inv ≫ homologyMap φ ≫ h₂.homologyIso.hom ≫ h₂.ι =
      h₁.i ≫ φ.τ₂ ≫ h₂.p := by
  dsimp only [LeftHomologyData.homologyIso, RightHomologyData.homologyIso,
    Iso.symm, Iso.trans, Iso.refl, leftHomologyIso, rightHomologyIso,
    leftHomologyMapIso', rightHomologyMapIso',
    LeftHomologyData.cyclesIso, RightHomologyData.opcyclesIso,
    LeftHomologyData.leftHomologyIso, RightHomologyData.rightHomologyIso,
    homologyMap, homologyMap']
  simp only [assoc, rightHomologyι_naturality', rightHomologyι_naturality'_assoc,
    leftHomologyπ_naturality'_assoc, HomologyData.comm_assoc, p_opcyclesMap'_assoc,
    id_τ₂, p_opcyclesMap', id_comp, cyclesMap'_i_assoc]


@[reassoc]
lemma π_homologyMap_ι [S₁.HasHomology] [S₂.HasHomology] (φ : S₁ ⟶ S₂) :
    S₁.homologyπ ≫ homologyMap φ ≫ S₂.homologyι = S₁.iCycles ≫ φ.τ₂ ≫ S₂.pOpcycles := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S₁.homologyπ (CategoryTheory.Category …
  -/
  simp only [homologyι_naturality, homology_π_ι_assoc, p_opcyclesMap]
  /-
    🎉 no goals
  -/


/-- The canonical isomorphism `S.op.homology ≅ Opposite.op S.homology` when a short
complex `S` has homology. -/
noncomputable def homologyOpIso [S.HasHomology] :
    S.op.homology ≅ Opposite.op S.homology :=
  S.op.leftHomologyIso.symm ≪≫ S.leftHomologyOpIso ≪≫ S.rightHomologyIso.symm.op


lemma homologyMap'_op : (homologyMap' φ h₁ h₂).op =
    h₂.iso.inv.op ≫ homologyMap' (opMap φ) h₂.op h₁.op ≫ h₁.iso.hom.op :=
  Quiver.Hom.unop_inj (by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.HomologyData
      h₂ : S₂.HomologyData
      ⊢ Eq (CategoryTheory.ShortComplex.homologyMap' φ h₁ h₂).op.unop (CategoryTheor …
    -/
    dsimp
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.HomologyData
      h₂ : S₂.HomologyData
      ⊢ Eq (CategoryTheory.ShortComplex.homologyMap' φ h₁ h₂) (CategoryTheory.Catego …
    -/
    have γ : HomologyMapData φ h₁ h₂ := default
    simp only [γ.homologyMap'_eq, γ.op.homologyMap'_eq, HomologyData.op_left,
      HomologyMapData.op_left, RightHomologyMapData.op_φH, Quiver.Hom.unop_op, assoc,
      ← γ.comm_assoc, Iso.hom_inv_id, comp_id])


lemma homologyMap_op [HasHomology S₁] [HasHomology S₂] :
    (homologyMap φ).op =
      (S₂.homologyOpIso).inv ≫ homologyMap (opMap φ) ≫ (S₁.homologyOpIso).hom := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    ⊢ Eq (CategoryTheory.ShortComplex.homologyMap φ).op (CategoryTheory.CategorySt …
  -/
  dsimp only [homologyMap, homologyOpIso]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    ⊢ Eq (CategoryTheory.ShortComplex.homologyMap' φ S₁.homologyData S₂.homologyDa …
  -/
  rw [homologyMap'_op]
  dsimp only [Iso.symm, Iso.trans, Iso.op, Iso.refl, rightHomologyIso, leftHomologyIso,
    leftHomologyOpIso, leftHomologyMapIso', rightHomologyMapIso',
    LeftHomologyData.leftHomologyIso, homologyMap']
  simp only [assoc, rightHomologyMap'_op, op_comp, ← leftHomologyMap'_comp_assoc, id_comp,
    opMap_id, comp_id, HomologyData.op_left]


@[reassoc]
lemma homologyOpIso_hom_naturality [S₁.HasHomology] [S₂.HasHomology] :
    homologyMap (opMap φ) ≫ (S₁.homologyOpIso).hom =
      S₂.homologyOpIso.hom ≫ (homologyMap φ).op := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.homology …
  -/
  simp [homologyMap_op]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma homologyOpIso_inv_naturality [S₁.HasHomology] [S₂.HasHomology] :
    (homologyMap φ).op ≫ (S₁.homologyOpIso).inv =
      S₂.homologyOpIso.inv ≫ homologyMap (opMap φ) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.homology …
  -/
  simp [homologyMap_op]
  /-
    🎉 no goals
  -/


/-- The natural isomorphism `(homologyFunctor C).op ≅ opFunctor C ⋙ homologyFunctor Cᵒᵖ`
which relates the homology in `C` and in `Cᵒᵖ`. -/
noncomputable def homologyFunctorOpNatIso [CategoryWithHomology C] :
    (homologyFunctor C).op ≅ opFunctor C ⋙ homologyFunctor Cᵒᵖ :=
  NatIso.ofComponents (fun S => S.unop.homologyOpIso.symm)
    (fun _ ↦ homologyOpIso_inv_naturality _)


lemma liftCycles_homologyπ_eq_zero_of_boundary [S.HasHomology]
    (k : A ⟶ S.X₂) (x : A ⟶ S.X₁) (hx : k = x ≫ S.f) :
                       /-
                         C : Type u
                         inst✝² : CategoryTheory.Category.{v, u} C
                         inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                         S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
                         φ : Quiver.Hom S₁ S₂
                         h₁ : S₁.HomologyData
                         h₂ : S₂.HomologyData
                         A : C
                         inst✝ : S.HasHomology
                         k : Quiver.Hom A S.X₂
                         x : Quiver.Hom A S.X₁
                         hx : Eq k (CategoryTheory.CategoryStruct.comp x S.f)
                         ⊢ Eq (CategoryTheory.CategoryStruct.comp k S.g) 0
                       -/
    S.liftCycles k (by rw [hx, assoc, S.zero, comp_zero]) ≫ S.homologyπ = 0 := by
                       /-
                         🎉 no goals
                       -/
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    A : C
    inst✝ : S.HasHomology
    k : Quiver.Hom A S.X₂
    x : Quiver.Hom A S.X₁
    hx : Eq k (CategoryTheory.CategoryStruct.comp x S.f)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.liftCycles k ⋯) S.homologyπ) 0
  -/
  dsimp only [homologyπ]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    A : C
    inst✝ : S.HasHomology
    k : Quiver.Hom A S.X₂
    x : Quiver.Hom A S.X₁
    hx : Eq k (CategoryTheory.CategoryStruct.comp x S.f)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.liftCycles k ⋯) (CategoryTheory.Ca …
  -/
  rw [S.liftCycles_leftHomologyπ_eq_zero_of_boundary_assoc k x hx, zero_comp]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma homologyι_descOpcycles_eq_zero_of_boundary [S.HasHomology]
    (k : S.X₂ ⟶ A) (x : S.X₃ ⟶ A) (hx : k = S.g ≫ x) :
                                       /-
                                         C : Type u
                                         inst✝² : CategoryTheory.Category.{v, u} C
                                         inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                         S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
                                         φ : Quiver.Hom S₁ S₂
                                         h₁ : S₁.HomologyData
                                         h₂ : S₂.HomologyData
                                         A : C
                                         inst✝ : S.HasHomology
                                         k : Quiver.Hom S.X₂ A
                                         x : Quiver.Hom S.X₃ A
                                         hx : Eq k (CategoryTheory.CategoryStruct.comp S.g x)
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp S.f k) 0
                                       -/
    S.homologyι ≫ S.descOpcycles k (by rw [hx, S.zero_assoc, zero_comp]) = 0 := by
                                       /-
                                         🎉 no goals
                                       -/
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    A : C
    inst✝ : S.HasHomology
    k : Quiver.Hom S.X₂ A
    x : Quiver.Hom S.X₃ A
    hx : Eq k (CategoryTheory.CategoryStruct.comp S.g x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.homologyι (S.descOpcycles k ⋯)) 0
  -/
  dsimp only [homologyι]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    A : C
    inst✝ : S.HasHomology
    k : Quiver.Hom S.X₂ A
    x : Quiver.Hom S.X₃ A
    hx : Eq k (CategoryTheory.CategoryStruct.comp S.g x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp S …
  -/
  rw [assoc, S.rightHomologyι_descOpcycles_π_eq_zero_of_boundary k x hx, comp_zero]
  /-
    🎉 no goals
  -/


lemma isIso_homologyMap_of_isIso_cyclesMap_of_epi {φ : S₁ ⟶ S₂}
    [S₁.HasHomology] [S₂.HasHomology] (h₁ : IsIso (cyclesMap φ)) (h₂ : Epi φ.τ₁) :
    IsIso (homologyMap φ) := by
  have h : S₂.toCycles ≫ inv (cyclesMap φ) ≫ S₁.homologyπ = 0 := by
    simp only [← cancel_epi φ.τ₁, ← toCycles_naturality_assoc,
      IsIso.hom_inv_id_assoc, toCycles_comp_homologyπ, comp_zero]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    h₁ : CategoryTheory.IsIso (CategoryTheory.ShortComplex.cyclesMap φ)
    h₂ : CategoryTheory.Epi φ.τ₁
    h : Eq (CategoryTheory.CategoryStruct.comp S₂.toCycles (CategoryTheory.Categor …
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ)
  -/
  have ⟨z, hz⟩ := CokernelCofork.IsColimit.desc' S₂.homologyIsCokernel _ h
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    h₁ : CategoryTheory.IsIso (CategoryTheory.ShortComplex.cyclesMap φ)
    h₂ : CategoryTheory.Epi φ.τ₁
    h : Eq (CategoryTheory.CategoryStruct.comp S₂.toCycles (CategoryTheory.Categor …
    z : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ S₂.homologyπ ⋯).pt S₁ …
    hz : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (C …
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ)
  -/
  dsimp at hz
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    h₁ : CategoryTheory.IsIso (CategoryTheory.ShortComplex.cyclesMap φ)
    h₂ : CategoryTheory.Epi φ.τ₁
    h : Eq (CategoryTheory.CategoryStruct.comp S₂.toCycles (CategoryTheory.Categor …
    z : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ S₂.homologyπ ⋯).pt S₁ …
    hz : Eq (CategoryTheory.CategoryStruct.comp S₂.homologyπ z) (CategoryTheory.Ca …
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ)
  -/
  refine ⟨⟨z, ?_, ?_⟩⟩
  · rw [← cancel_epi S₁.homologyπ, homologyπ_naturality_assoc, hz,
      IsIso.hom_inv_id_assoc, comp_id]
  · rw [← cancel_epi S₂.homologyπ, reassoc_of% hz, homologyπ_naturality,
      IsIso.inv_hom_id_assoc, comp_id]


lemma isIso_homologyMap_of_isIso_opcyclesMap_of_mono {φ : S₁ ⟶ S₂}
    [S₁.HasHomology] [S₂.HasHomology] (h₁ : IsIso (opcyclesMap φ)) (h₂ : Mono φ.τ₃) :
    IsIso (homologyMap φ) := by
  have h : (S₂.homologyι ≫ inv (opcyclesMap φ)) ≫ S₁.fromOpcycles = 0 := by
    simp only [← cancel_mono φ.τ₃, zero_comp, assoc, ← fromOpcycles_naturality,
      IsIso.inv_hom_id_assoc, homologyι_comp_fromOpcycles]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    h₁ : CategoryTheory.IsIso (CategoryTheory.ShortComplex.opcyclesMap φ)
    h₂ : CategoryTheory.Mono φ.τ₃
    h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ)
  -/
  have ⟨z, hz⟩ := KernelFork.IsLimit.lift' S₁.homologyIsKernel _ h
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    h₁ : CategoryTheory.IsIso (CategoryTheory.ShortComplex.opcyclesMap φ)
    h₂ : CategoryTheory.Mono φ.τ₃
    h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
    z : Quiver.Hom S₂.homology (CategoryTheory.Limits.KernelFork.ofι S₁.homologyι  …
    hz : Eq (CategoryTheory.CategoryStruct.comp z (CategoryTheory.Limits.Fork.ι (C …
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ)
  -/
  dsimp at hz
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    h₁ : CategoryTheory.IsIso (CategoryTheory.ShortComplex.opcyclesMap φ)
    h₂ : CategoryTheory.Mono φ.τ₃
    h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
    z : Quiver.Hom S₂.homology (CategoryTheory.Limits.KernelFork.ofι S₁.homologyι  …
    hz : Eq (CategoryTheory.CategoryStruct.comp z S₁.homologyι) (CategoryTheory.Ca …
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ)
  -/
  refine ⟨⟨z, ?_, ?_⟩⟩
  · rw [← cancel_mono S₁.homologyι, id_comp, assoc, hz, homologyι_naturality_assoc,
      IsIso.hom_inv_id, comp_id]
  · rw [← cancel_mono S₂.homologyι, assoc, homologyι_naturality, reassoc_of% hz,
      IsIso.inv_hom_id, comp_id, id_comp]


lemma isZero_homology_of_isZero_X₂ (hS : IsZero S.X₂) [S.HasHomology] :
    IsZero S.homology :=
  IsZero.of_iso hS (HomologyData.ofZeros S (hS.eq_of_tgt _ _)
    (hS.eq_of_src _ _)).left.homologyIso


lemma isIso_homologyπ (hf : S.f = 0) [S.HasHomology] :
    IsIso S.homologyπ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    hf : Eq S.f 0
    inst✝ : S.HasHomology
    ⊢ CategoryTheory.IsIso S.homologyπ
  -/
  have := S.isIso_leftHomologyπ hf
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    hf : Eq S.f 0
    inst✝ : S.HasHomology
    this : CategoryTheory.IsIso S.leftHomologyπ
    ⊢ CategoryTheory.IsIso S.homologyπ
  -/
  dsimp only [homologyπ]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    hf : Eq S.f 0
    inst✝ : S.HasHomology
    this : CategoryTheory.IsIso S.leftHomologyπ
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp S.leftHomologyπ S.l …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma isIso_homologyι (hg : S.g = 0) [S.HasHomology] :
    IsIso S.homologyι := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    hg : Eq S.g 0
    inst✝ : S.HasHomology
    ⊢ CategoryTheory.IsIso S.homologyι
  -/
  have := S.isIso_rightHomologyι hg
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    hg : Eq S.g 0
    inst✝ : S.HasHomology
    this : CategoryTheory.IsIso S.rightHomologyι
    ⊢ CategoryTheory.IsIso S.homologyι
  -/
  dsimp only [homologyι]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    hg : Eq S.g 0
    inst✝ : S.HasHomology
    this : CategoryTheory.IsIso S.rightHomologyι
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp S.rightHomologyIso. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The canonical isomorphism `S.cycles ≅ S.homology` when `S.f = 0`. -/
@[simps! hom]
noncomputable def asIsoHomologyπ (hf : S.f = 0) [S.HasHomology] :
    S.cycles ≅ S.homology := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.HomologyData
    h₂ : S₂.HomologyData
    A : C
    hf : Eq S.f 0
    inst✝ : S.HasHomology
    ⊢ CategoryTheory.Iso S.cycles S.homology
  -/
  have := S.isIso_homologyπ hf
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.HomologyData
    h₂ : S₂.HomologyData
    A : C
    hf : Eq S.f 0
    inst✝ : S.HasHomology
    this : CategoryTheory.IsIso S.homologyπ
    ⊢ CategoryTheory.Iso S.cycles S.homology
  -/
  exact asIso S.homologyπ
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma asIsoHomologyπ_inv_comp_homologyπ (hf : S.f = 0) [S.HasHomology] :
    (S.asIsoHomologyπ hf).inv ≫ S.homologyπ = 𝟙 _ := Iso.inv_hom_id _


@[reassoc (attr := simp)]
lemma homologyπ_comp_asIsoHomologyπ_inv (hf : S.f = 0) [S.HasHomology] :
    S.homologyπ ≫ (S.asIsoHomologyπ hf).inv = 𝟙 _ := (S.asIsoHomologyπ hf).hom_inv_id


/-- The canonical isomorphism `S.homology ≅ S.opcycles` when `S.g = 0`. -/
@[simps! hom]
noncomputable def asIsoHomologyι (hg : S.g = 0) [S.HasHomology] :
    S.homology ≅ S.opcycles := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.HomologyData
    h₂ : S₂.HomologyData
    A : C
    hg : Eq S.g 0
    inst✝ : S.HasHomology
    ⊢ CategoryTheory.Iso S.homology S.opcycles
  -/
  have := S.isIso_homologyι hg
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.HomologyData
    h₂ : S₂.HomologyData
    A : C
    hg : Eq S.g 0
    inst✝ : S.HasHomology
    this : CategoryTheory.IsIso S.homologyι
    ⊢ CategoryTheory.Iso S.homology S.opcycles
  -/
  exact asIso S.homologyι
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma asIsoHomologyι_inv_comp_homologyι (hg : S.g = 0) [S.HasHomology] :
    (S.asIsoHomologyι hg).inv ≫ S.homologyι = 𝟙 _ := Iso.inv_hom_id _


@[reassoc (attr := simp)]
lemma homologyι_comp_asIsoHomologyι_inv (hg : S.g = 0) [S.HasHomology] :
    S.homologyι ≫ (S.asIsoHomologyι hg).inv  = 𝟙 _ := (S.asIsoHomologyι hg).hom_inv_id


lemma mono_homologyMap_of_mono_opcyclesMap'
    [S₁.HasHomology] [S₂.HasHomology] (h : Mono (opcyclesMap φ)) :
    Mono (homologyMap φ) := by
  have : Mono (homologyMap φ ≫ S₂.homologyι) := by
    rw [homologyι_naturality φ]
    apply mono_comp
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    h : CategoryTheory.Mono (CategoryTheory.ShortComplex.opcyclesMap φ)
    this : CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp (CategoryTheory …
    ⊢ CategoryTheory.Mono (CategoryTheory.ShortComplex.homologyMap φ)
  -/
  exact mono_of_mono (homologyMap φ) S₂.homologyι
  /-
    🎉 no goals
  -/


instance mono_homologyMap_of_mono_opcyclesMap
    [S₁.HasHomology] [S₂.HasHomology] [Mono (opcyclesMap φ)] :
    Mono (homologyMap φ) :=
  mono_homologyMap_of_mono_opcyclesMap' φ inferInstance


lemma epi_homologyMap_of_epi_cyclesMap'
    [S₁.HasHomology] [S₂.HasHomology] (h : Epi (cyclesMap φ)) :
    Epi (homologyMap φ) := by
  have : Epi (S₁.homologyπ ≫ homologyMap φ) := by
    rw [homologyπ_naturality φ]
    apply epi_comp
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    h : CategoryTheory.Epi (CategoryTheory.ShortComplex.cyclesMap φ)
    this : CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp S₁.homologyπ (Ca …
    ⊢ CategoryTheory.Epi (CategoryTheory.ShortComplex.homologyMap φ)
  -/
  exact epi_of_epi S₁.homologyπ (homologyMap φ)
  /-
    🎉 no goals
  -/


instance epi_homologyMap_of_epi_cyclesMap
    [S₁.HasHomology] [S₂.HasHomology] [Epi (cyclesMap φ)] :
    Epi (homologyMap φ) :=
  epi_homologyMap_of_epi_cyclesMap' φ inferInstance


