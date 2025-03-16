/-- A type synonym for endowing a given type with a `CStarModule` structure. This has the scoped
notation `C⋆ᵐᵒᵈ`.

Note: because the C⋆-algebra `A` over which `E` is a `CStarModule` is listed as an `outParam` in
that class, we don't pass it as an unused argument to `WithCStarModule`, unlike the `p` parameter
in `WithLp`, which can vary. -/
def WithCStarModule (E : Type*) := E


@[inherit_doc]
scoped notation "C⋆ᵐᵒᵈ" => WithCStarModule


/-- The canonical equivalence between `WithCStarModule E` and `E`. This should always be used to
convert back and forth between the representations. -/
def equiv : WithCStarModule E ≃ E := Equiv.refl _


instance instNontrivial [Nontrivial E] : Nontrivial (WithCStarModule E) := ‹Nontrivial E›

instance instInhabited [Inhabited E] : Inhabited (WithCStarModule E) := ‹Inhabited E›

instance instNonempty [Nonempty E] : Nonempty (WithCStarModule E) := ‹Nonempty E›

instance instUnique [Unique E] : Unique (WithCStarModule E) := ‹Unique E›


instance instAddCommGroup [AddCommGroup E] : AddCommGroup (WithCStarModule E) := ‹AddCommGroup E›

instance instSMul {R : Type*} [SMul R E] : SMul R (WithCStarModule E) := ‹SMul R E›

instance instModule {R : Type*} [Semiring R] [AddCommGroup E] [Module R E] :
    Module R (WithCStarModule E) :=
  ‹Module R E›


instance instIsScalarTower [SMul R R'] [SMul R E] [SMul R' E]
    [IsScalarTower R R' E] : IsScalarTower R R' (WithCStarModule E) :=
  ‹IsScalarTower R R' E›


instance instSMulCommClass [SMul R E] [SMul R' E] [SMulCommClass R R' E] :
    SMulCommClass R R' (WithCStarModule E) :=
  ‹SMulCommClass R R' E›


instance instModuleFinite [Semiring R] [AddCommGroup E] [Module R E] [Module.Finite R E] :
    Module.Finite R (WithCStarModule E) :=
  ‹Module.Finite R E›



@[simp]
theorem equiv_zero : equiv E 0 = 0 :=
  rfl


@[simp]
theorem equiv_symm_zero : (equiv E).symm 0 = 0 :=
  rfl


@[simp]
theorem equiv_add : equiv E (x + y) = equiv E x + equiv E y :=
  rfl


@[simp]
theorem equiv_symm_add :
    (equiv E).symm (x' + y') = (equiv E).symm x' + (equiv E).symm y' :=
  rfl


@[simp]
theorem equiv_sub : equiv E (x - y) = equiv E x - equiv E y :=
  rfl


@[simp]
theorem equiv_symm_sub :
    (equiv E).symm (x' - y') = (equiv E).symm x' - (equiv E).symm y' :=
  rfl


@[simp]
theorem equiv_neg : equiv E (-x) = -equiv E x :=
  rfl


@[simp]
theorem equiv_symm_neg : (equiv E).symm (-x') = -(equiv E).symm x' :=
  rfl


@[simp]
theorem equiv_smul : equiv E (c • x) = c • equiv E x :=
  rfl


@[simp]
theorem equiv_symm_smul : (equiv E).symm (c • x') = c • (equiv E).symm x' :=
  rfl


/-- `WithCStarModule.equiv` as a linear equivalence. -/
@[simps (config := .asFn)]
def linearEquiv [Semiring R] [AddCommGroup E] [Module R E] : C⋆ᵐᵒᵈ E ≃ₗ[R] E :=
  { LinearEquiv.refl _ _ with
    toFun := equiv _
    invFun := (equiv _).symm }


instance [u : UniformSpace E] : UniformSpace (C⋆ᵐᵒᵈ E) := u.comap <| equiv E


instance [Bornology E] : Bornology (C⋆ᵐᵒᵈ E) := Bornology.induced <| equiv E


/-- `WithCStarModule.equiv` as a uniform equivalence between `C⋆ᵐᵒᵈ E` and `E`. -/
def uniformEquiv [UniformSpace E] : C⋆ᵐᵒᵈ E ≃ᵤ E :=
  equiv E |>.toUniformEquivOfIsUniformInducing ⟨rfl⟩


instance [UniformSpace E] [CompleteSpace E] : CompleteSpace (C⋆ᵐᵒᵈ E) :=
  uniformEquiv.completeSpace_iff.mpr inferInstance


@[simp]
theorem zero_fst : (0 : C⋆ᵐᵒᵈ (E × F)).fst = 0 :=
  rfl


@[simp]
theorem zero_snd : (0 : C⋆ᵐᵒᵈ (E × F)).snd = 0 :=
  rfl


@[simp]
theorem add_fst : (x + y).fst = x.fst + y.fst :=
  rfl


@[simp]
theorem add_snd : (x + y).snd = x.snd + y.snd :=
  rfl


@[simp]
theorem sub_fst : (x - y).fst = x.fst - y.fst :=
  rfl


@[simp]
theorem sub_snd : (x - y).snd = x.snd - y.snd :=
  rfl


@[simp]
theorem neg_fst : (-x).fst = -x.fst :=
  rfl


@[simp]
theorem neg_snd : (-x).snd = -x.snd :=
  rfl


@[simp]
theorem smul_fst : (c • x).fst = c • x.fst :=
  rfl


@[simp]
theorem smul_snd : (c • x).snd = c • x.snd :=
  rfl


@[simp]
theorem equiv_fst (x : C⋆ᵐᵒᵈ (E × F)) : (equiv (E × F) x).fst = x.fst :=
  rfl


@[simp]
theorem equiv_snd (x : C⋆ᵐᵒᵈ (E × F)) : (equiv (E × F) x).snd = x.snd :=
  rfl


@[simp]
theorem equiv_symm_fst (x : E × F) : ((equiv (E × F)).symm x).fst = x.fst :=
  rfl


@[simp]
theorem equiv_symm_snd (x : E × F) : ((equiv (E × F)).symm x).snd = x.snd :=
  rfl


instance {ι : Type*} (E : ι → Type*) : CoeFun (C⋆ᵐᵒᵈ (Π i, E i)) (fun _ ↦ Π i, E i) where
  coe := WithCStarModule.equiv _


@[ext]
protected theorem ext {ι : Type*} {E : ι → Type*} {x y : C⋆ᵐᵒᵈ (Π i, E i)}
    (h : ∀ i, x i = y i) : x = y :=
  funext h


@[simp]
theorem zero_apply : (0 : C⋆ᵐᵒᵈ (Π i, E i)) i = 0 :=
  rfl


@[simp]
theorem add_apply : (x + y) i = x i + y i :=
  rfl


@[simp]
theorem sub_apply : (x - y) i = x i - y i :=
  rfl


@[simp]
theorem neg_apply : (-x) i = -x i :=
  rfl


@[simp]
theorem smul_apply : (c • x) i = c • x i :=
  rfl


@[simp]
theorem equiv_pi_apply (i : ι) : equiv _ x i = x i :=
  rfl


@[simp]
theorem equiv_symm_pi_apply (x : ∀ i, E i) (i : ι) :
    (WithCStarModule.equiv _).symm x i = x i :=
  rfl


