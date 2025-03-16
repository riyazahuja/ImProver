variable (I I') in
/-- Bundled `n` times continuously differentiable maps. -/
def ContMDiffMap :=
  { f : M → M' // ContMDiff I I' n f }


@[deprecated (since := "024-11-21")] alias SmoothMap := ContMDiffMap


@[inherit_doc]
scoped[Manifold] notation "C^" n "⟮" I ", " M "; " I' ", " M' "⟯" => ContMDiffMap I I' M M' n


@[inherit_doc]
scoped[Manifold]
  notation "C^" n "⟮" I ", " M "; " k "⟯" => ContMDiffMap I (modelWithCornersSelf k k) M k n


local notation "∞" => (⊤ : ℕ∞)



instance instFunLike : FunLike C^n⟮I, M; I', M'⟯ M M' where
  coe := Subtype.val
  coe_injective' := Subtype.coe_injective


protected theorem contMDiff (f : C^n⟮I, M; I', M'⟯) : ContMDiff I I' n f :=
  f.prop


@[deprecated (since := "2024-11-20")] alias smooth := ContMDiffMap.contMDiff

-- Porting note: use generic instance instead
-- instance : Coe C^n⟮I, M; I', M'⟯ C(M, M') :=
--   ⟨fun f => ⟨f, f.contMDiff.continuous⟩⟩


@[simp]
theorem coeFn_mk (f : M → M') (hf : ContMDiff I I' n f) :
    DFunLike.coe (F := C^n⟮I, M; I', M'⟯) ⟨f, hf⟩ = f :=
  rfl


theorem coe_injective ⦃f g : C^n⟮I, M; I', M'⟯⦄ (h : (f : M → M') = g) : f = g :=
  DFunLike.ext' h


@[ext]
theorem ext (h : ∀ x, f x = g x) : f = g := DFunLike.ext _ _ h


instance : ContinuousMapClass C^n⟮I, M; I', M'⟯ M M' where
  map_continuous f := f.contMDiff.continuous


/-- The identity as a smooth map. -/
nonrec def id : C^n⟮I, M; I, M⟯ :=
  ⟨id, contMDiff_id⟩


/-- The composition of smooth maps, as a smooth map. -/
def comp (f : C^n⟮I', M'; I'', M''⟯) (g : C^n⟮I, M; I', M'⟯) : C^n⟮I, M; I'', M''⟯ where
  val a := f (g a)
  property := f.contMDiff.comp g.contMDiff


@[simp]
theorem comp_apply (f : C^n⟮I', M'; I'', M''⟯) (g : C^n⟮I, M; I', M'⟯) (x : M) :
    f.comp g x = f (g x) :=
  rfl


instance [Inhabited M'] : Inhabited C^n⟮I, M; I', M'⟯ :=
  ⟨⟨fun _ => default, contMDiff_const⟩⟩


/-- Constant map as a smooth map -/
def const (y : M') : C^n⟮I, M; I', M'⟯ :=
  ⟨fun _ => y, contMDiff_const⟩


/-- The first projection of a product, as a smooth map. -/
def fst : C^n⟮I.prod I', M × M'; I, M⟯ :=
  ⟨Prod.fst, contMDiff_fst⟩


/-- The second projection of a product, as a smooth map. -/
def snd : C^n⟮I.prod I', M × M'; I', M'⟯ :=
  ⟨Prod.snd, contMDiff_snd⟩


/-- Given two smooth maps `f` and `g`, this is the smooth map `x ↦ (f x, g x)`. -/
def prodMk (f : C^n⟮J, N; I, M⟯) (g : C^n⟮J, N; I', M'⟯) : C^n⟮J, N; I.prod I', M × M'⟯ :=
  ⟨fun x => (f x, g x), f.2.prod_mk g.2⟩


instance ContinuousLinearMap.hasCoeToContMDiffMap :
    Coe (E →L[𝕜] E') C^n⟮𝓘(𝕜, E), E; 𝓘(𝕜, E'), E'⟯ :=
  ⟨fun f => ⟨f, f.contMDiff⟩⟩

