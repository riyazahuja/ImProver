/-- A function between topological spaces is spectral if it is continuous and the preimage of every
compact open set is compact open. -/
@[stacks 005A, stacks 08YG]
structure IsSpectralMap (f : α → β) extends Continuous f : Prop where
  /-- A function between topological spaces is spectral if it is continuous and the preimage of
   every compact open set is compact open. -/
  isCompact_preimage_of_isOpen ⦃s : Set β⦄ : IsOpen s → IsCompact s → IsCompact (f ⁻¹' s)


theorem IsCompact.preimage_of_isOpen (hf : IsSpectralMap f) (h₀ : IsCompact s) (h₁ : IsOpen s) :
    IsCompact (f ⁻¹' s) :=
  hf.isCompact_preimage_of_isOpen h₁ h₀


theorem IsSpectralMap.continuous {f : α → β} (hf : IsSpectralMap f) : Continuous f :=
  hf.toContinuous


theorem isSpectralMap_id : IsSpectralMap (@id α) :=
  ⟨continuous_id, fun _s _ => id⟩


@[stacks 005B]
theorem IsSpectralMap.comp {f : β → γ} {g : α → β} (hf : IsSpectralMap f) (hg : IsSpectralMap g) :
    IsSpectralMap (f ∘ g) :=
  ⟨hf.continuous.comp hg.continuous, fun _s hs₀ hs₁ =>
    ((hs₁.preimage_of_isOpen hf hs₀).preimage_of_isOpen hg) (hs₀.preimage hf.continuous)⟩


/-- The type of spectral maps from `α` to `β`. -/
structure SpectralMap (α β : Type*) [TopologicalSpace α] [TopologicalSpace β] where
  /-- function between topological spaces -/
  toFun : α → β
  /-- proof that `toFun` is a spectral map -/
  spectral' : IsSpectralMap toFun


/-- `SpectralMapClass F α β` states that `F` is a type of spectral maps.

You should extend this class when you extend `SpectralMap`. -/
class SpectralMapClass (F α β : Type*) [TopologicalSpace α] [TopologicalSpace β]
    [FunLike F α β] : Prop where
  /-- statement that `F` is a type of spectral maps -/
  map_spectral (f : F) : IsSpectralMap f


attribute [simp] map_spectral

-- See note [lower instance priority]

instance (priority := 100) SpectralMapClass.toContinuousMapClass [TopologicalSpace α]
    [TopologicalSpace β] [FunLike F α β] [SpectralMapClass F α β] : ContinuousMapClass F α β :=
  { ‹SpectralMapClass F α β› with map_continuous := fun f => (map_spectral f).continuous }


instance [TopologicalSpace α] [TopologicalSpace β] [FunLike F α β] [SpectralMapClass F α β] :
    CoeTC F (SpectralMap α β) :=
  ⟨fun f => ⟨_, map_spectral f⟩⟩


/-- Reinterpret a `SpectralMap` as a `ContinuousMap`. -/
def toContinuousMap (f : SpectralMap α β) : ContinuousMap α β :=
  ⟨_, f.spectral'.continuous⟩


instance instFunLike : FunLike (SpectralMap α β) α β where
  coe := SpectralMap.toFun
                             /-
                               F : Type u_1
                               α : Type u_2
                               β : Type u_3
                               γ : Type u_4
                               δ : Type u_5
                               inst✝³ : TopologicalSpace α
                               inst✝² : TopologicalSpace β
                               inst✝¹ : TopologicalSpace γ
                               inst✝ : TopologicalSpace δ
                               f g : SpectralMap α β
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


instance : SpectralMapClass (SpectralMap α β) α β where
  map_spectral f := f.spectral'


@[simp]
theorem toFun_eq_coe {f : SpectralMap α β} : f.toFun = (f : α → β) :=
  rfl


@[ext]
theorem ext {f g : SpectralMap α β} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


/-- Copy of a `SpectralMap` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (f : SpectralMap α β) (f' : α → β) (h : f' = f) : SpectralMap α β :=
  ⟨f', h.symm.subst f.spectral'⟩


@[simp]
theorem coe_copy (f : SpectralMap α β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : SpectralMap α β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- `id` as a `SpectralMap`. -/
protected def id : SpectralMap α α :=
  ⟨id, isSpectralMap_id⟩


instance : Inhabited (SpectralMap α α) :=
  ⟨SpectralMap.id α⟩


@[simp]
theorem coe_id : ⇑(SpectralMap.id α) = id :=
  rfl


@[simp]
theorem id_apply (a : α) : SpectralMap.id α a = a :=
  rfl


/-- Composition of `SpectralMap`s as a `SpectralMap`. -/
def comp (f : SpectralMap β γ) (g : SpectralMap α β) : SpectralMap α γ :=
  ⟨f.toContinuousMap.comp g.toContinuousMap, f.spectral'.comp g.spectral'⟩


@[simp]
theorem coe_comp (f : SpectralMap β γ) (g : SpectralMap α β) : (f.comp g : α → γ) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : SpectralMap β γ) (g : SpectralMap α β) (a : α) : (f.comp g) a = f (g a) :=
  rfl


theorem coe_comp_continuousMap (f : SpectralMap β γ) (g : SpectralMap α β) :
    f ∘ g = (f : ContinuousMap β γ) ∘ (g : ContinuousMap α β) :=
   rfl


@[simp]
theorem coe_comp_continuousMap' (f : SpectralMap β γ) (g : SpectralMap α β) :
    (f.comp g : ContinuousMap α γ) = (f : ContinuousMap β γ).comp g :=
  rfl


@[simp]
theorem comp_assoc (f : SpectralMap γ δ) (g : SpectralMap β γ) (h : SpectralMap α β) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp]
theorem comp_id (f : SpectralMap α β) : f.comp (SpectralMap.id α) = f :=
  ext fun _a => rfl


@[simp]
theorem id_comp (f : SpectralMap α β) : (SpectralMap.id β).comp f = f :=
  ext fun _a => rfl


@[simp]
theorem cancel_right {g₁ g₂ : SpectralMap β γ} {f : SpectralMap α β} (hf : Surjective f) :
    g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => ext <| hf.forall.2 <| DFunLike.ext_iff.1 h,
   fun a => of_eq (congrFun (congrArg comp a) f)⟩


@[simp]
theorem cancel_left {g : SpectralMap β γ} {f₁ f₂ : SpectralMap α β} (hg : Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    γ : Type u_4
                                    inst✝² : TopologicalSpace α
                                    inst✝¹ : TopologicalSpace β
                                    inst✝ : TopologicalSpace γ
                                    g : SpectralMap β γ
                                    f₁ f₂ : SpectralMap α β
                                    hg : Function.Injective ⇑g
                                    h : Eq (g.comp f₁) (g.comp f₂)
                                    a : α
                                    ⊢ Eq (g (f₁ a)) (g (f₂ a))
                                  -/
  ⟨fun h => ext fun a => hg <| by rw [← comp_apply, h, comp_apply], congr_arg _⟩
                                  /-
                                    🎉 no goals
                                  -/


