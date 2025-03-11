/-- The normed group hom induced between completions. -/
def NormedAddGroupHom.completion (f : NormedAddGroupHom G H) :
    NormedAddGroupHom (Completion G) (Completion H) :=
  .ofLipschitz (f.toAddMonoidHom.completion f.continuous) f.lipschitz.completion_map


theorem NormedAddGroupHom.completion_def (f : NormedAddGroupHom G H) (x : Completion G) :
    f.completion x = Completion.map f x :=
  rfl


@[simp]
theorem NormedAddGroupHom.completion_coe_to_fun (f : NormedAddGroupHom G H) :
    (f.completion : Completion G → Completion H) = Completion.map f := rfl

-- Porting note: `@[simp]` moved to the next lemma

theorem NormedAddGroupHom.completion_coe (f : NormedAddGroupHom G H) (g : G) :
    f.completion g = f g :=
  Completion.map_coe f.uniformContinuous _


@[simp]
theorem NormedAddGroupHom.completion_coe' (f : NormedAddGroupHom G H) (g : G) :
    Completion.map f g = f g :=
  f.completion_coe g


/-- Completion of normed group homs as a normed group hom. -/
@[simps]
def normedAddGroupHomCompletionHom :
    NormedAddGroupHom G H →+ NormedAddGroupHom (Completion G) (Completion H) where
  toFun := NormedAddGroupHom.completion
  map_zero' := toAddMonoidHom_injective AddMonoidHom.completion_zero
  map_add' f g := toAddMonoidHom_injective <|
    f.toAddMonoidHom.completion_add g.toAddMonoidHom f.continuous g.continuous


@[simp]
theorem NormedAddGroupHom.completion_id :
    (NormedAddGroupHom.id G).completion = NormedAddGroupHom.id (Completion G) := by
  /-
    G : Type u_1
    inst✝ : SeminormedAddCommGroup G
    ⊢ Eq (NormedAddGroupHom.id G).completion (NormedAddGroupHom.id (UniformSpace.C …
  -/
  ext x
  /-
    case H
    G : Type u_1
    inst✝ : SeminormedAddCommGroup G
    x : UniformSpace.Completion G
    ⊢ Eq ((NormedAddGroupHom.id G).completion x) ((NormedAddGroupHom.id (UniformSp …
  -/
  rw [NormedAddGroupHom.completion_def, NormedAddGroupHom.coe_id, Completion.map_id]
  /-
    case H
    G : Type u_1
    inst✝ : SeminormedAddCommGroup G
    x : UniformSpace.Completion G
    ⊢ Eq (_root_.id x) ((NormedAddGroupHom.id (UniformSpace.Completion G)) x)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem NormedAddGroupHom.completion_comp (f : NormedAddGroupHom G H) (g : NormedAddGroupHom H K) :
    g.completion.comp f.completion = (g.comp f).completion := by
  /-
    G : Type u_1
    inst✝² : SeminormedAddCommGroup G
    H : Type u_2
    inst✝¹ : SeminormedAddCommGroup H
    K : Type u_3
    inst✝ : SeminormedAddCommGroup K
    f : NormedAddGroupHom G H
    g : NormedAddGroupHom H K
    ⊢ Eq (g.completion.comp f.completion) (g.comp f).completion
  -/
  ext x
  rw [NormedAddGroupHom.coe_comp, NormedAddGroupHom.completion_def,
    NormedAddGroupHom.completion_coe_to_fun, NormedAddGroupHom.completion_coe_to_fun,
    Completion.map_comp g.uniformContinuous f.uniformContinuous]
  /-
    case H
    G : Type u_1
    inst✝² : SeminormedAddCommGroup G
    H : Type u_2
    inst✝¹ : SeminormedAddCommGroup H
    K : Type u_3
    inst✝ : SeminormedAddCommGroup K
    f : NormedAddGroupHom G H
    g : NormedAddGroupHom H K
    x : UniformSpace.Completion G
    ⊢ Eq (UniformSpace.Completion.map (Function.comp ⇑g ⇑f) x) (UniformSpace.Compl …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem NormedAddGroupHom.completion_neg (f : NormedAddGroupHom G H) :
    (-f).completion = -f.completion :=
  map_neg (normedAddGroupHomCompletionHom : NormedAddGroupHom G H →+ _) f


theorem NormedAddGroupHom.completion_add (f g : NormedAddGroupHom G H) :
    (f + g).completion = f.completion + g.completion :=
  normedAddGroupHomCompletionHom.map_add f g


theorem NormedAddGroupHom.completion_sub (f g : NormedAddGroupHom G H) :
    (f - g).completion = f.completion - g.completion :=
  map_sub (normedAddGroupHomCompletionHom : NormedAddGroupHom G H →+ _) f g


@[simp]
theorem NormedAddGroupHom.zero_completion : (0 : NormedAddGroupHom G H).completion = 0 :=
  normedAddGroupHomCompletionHom.map_zero


/-- The map from a normed group to its completion, as a normed group hom. -/
@[simps] -- Porting note: added `@[simps]`
def NormedAddCommGroup.toCompl : NormedAddGroupHom G (Completion G) where
  toFun := (↑)
  map_add' := Completion.toCompl.map_add
                   /-
                     G : Type u_1
                     inst✝² : SeminormedAddCommGroup G
                     H : Type u_2
                     inst✝¹ : SeminormedAddCommGroup H
                     K : Type u_3
                     inst✝ : SeminormedAddCommGroup K
                     ⊢ ∀ (v : G), LE.le (Norm.norm (↑G v)) (HMul.hMul 1 (Norm.norm v))
                   -/
  bound' := ⟨1, by simp [le_refl]⟩
                   /-
                     🎉 no goals
                   -/


theorem NormedAddCommGroup.norm_toCompl (x : G) : ‖toCompl x‖ = ‖x‖ :=
  Completion.norm_coe x


theorem NormedAddCommGroup.denseRange_toCompl : DenseRange (toCompl : G → Completion G) :=
  Completion.isDenseInducing_coe.dense


@[simp]
theorem NormedAddGroupHom.completion_toCompl (f : NormedAddGroupHom G H) :
                                                     /-
                                                       G : Type u_1
                                                       inst✝¹ : SeminormedAddCommGroup G
                                                       H : Type u_2
                                                       inst✝ : SeminormedAddCommGroup H
                                                       f : NormedAddGroupHom G H
                                                       ⊢ Eq (f.completion.comp NormedAddCommGroup.toCompl) (NormedAddCommGroup.toComp …
                                                     -/
    f.completion.comp toCompl = toCompl.comp f := by ext x; simp
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem NormedAddGroupHom.norm_completion (f : NormedAddGroupHom G H) : ‖f.completion‖ = ‖f‖ :=
  le_antisymm (ofLipschitz_norm_le _ _) <| opNorm_le_bound _ (norm_nonneg _) fun x => by
    /-
      G : Type u_1
      inst✝¹ : SeminormedAddCommGroup G
      H : Type u_2
      inst✝ : SeminormedAddCommGroup H
      f : NormedAddGroupHom G H
      x : G
      ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (Norm.norm f.completion) (Norm.norm x))
    -/
    simpa using f.completion.le_opNorm x
    /-
      🎉 no goals
    -/


theorem NormedAddGroupHom.ker_le_ker_completion (f : NormedAddGroupHom G H) :
    (toCompl.comp <| incl f.ker).range ≤ f.completion.ker := by
  /-
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    H : Type u_2
    inst✝ : SeminormedAddCommGroup H
    f : NormedAddGroupHom G H
    ⊢ LE.le (NormedAddCommGroup.toCompl.comp (NormedAddGroupHom.incl f.ker)).range …
  -/
  rintro _ ⟨⟨g, h₀ : f g = 0⟩, rfl⟩
  /-
    case intro.mk
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    H : Type u_2
    inst✝ : SeminormedAddCommGroup H
    f : NormedAddGroupHom G H
    g : G
    h₀ : Eq (f g) 0
    ⊢ Membership.mem f.completion.ker ((NormedAddCommGroup.toCompl.comp (NormedAdd …
  -/
  simp [h₀, mem_ker, Completion.coe_zero]
  /-
    🎉 no goals
  -/


theorem NormedAddGroupHom.ker_completion {f : NormedAddGroupHom G H} {C : ℝ}
    (h : f.SurjectiveOnWith f.range C) :
    (f.completion.ker : Set <| Completion G) = closure (toCompl.comp <| incl f.ker).range := by
  /-
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    H : Type u_2
    inst✝ : SeminormedAddCommGroup H
    f : NormedAddGroupHom G H
    C : Real
    h : f.SurjectiveOnWith f.range C
    ⊢ Eq (↑f.completion.ker) (closure ↑(NormedAddCommGroup.toCompl.comp (NormedAdd …
  -/
  refine le_antisymm ?_ (closure_minimal f.ker_le_ker_completion f.completion.isClosed_ker)
  /-
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    H : Type u_2
    inst✝ : SeminormedAddCommGroup H
    f : NormedAddGroupHom G H
    C : Real
    h : f.SurjectiveOnWith f.range C
    ⊢ LE.le (↑f.completion.ker) (closure ↑(NormedAddCommGroup.toCompl.comp (Normed …
  -/
  rintro hatg (hatg_in : f.completion hatg = 0)
  /-
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    H : Type u_2
    inst✝ : SeminormedAddCommGroup H
    f : NormedAddGroupHom G H
    C : Real
    h : f.SurjectiveOnWith f.range C
    hatg : UniformSpace.Completion G
    hatg_in : Eq (f.completion hatg) 0
    ⊢ Membership.mem (closure ↑(NormedAddCommGroup.toCompl.comp (NormedAddGroupHom …
  -/
  rw [SeminormedAddCommGroup.mem_closure_iff]
  /-
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    H : Type u_2
    inst✝ : SeminormedAddCommGroup H
    f : NormedAddGroupHom G H
    C : Real
    h : f.SurjectiveOnWith f.range C
    hatg : UniformSpace.Completion G
    hatg_in : Eq (f.completion hatg) 0
    ⊢ ∀ (ε : Real), LT.lt 0 ε → Exists fun b => And (Membership.mem (↑(NormedAddCo …
  -/
  intro ε ε_pos
  /-
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    H : Type u_2
    inst✝ : SeminormedAddCommGroup H
    f : NormedAddGroupHom G H
    C : Real
    h : f.SurjectiveOnWith f.range C
    hatg : UniformSpace.Completion G
    hatg_in : Eq (f.completion hatg) 0
    ε : Real
    ε_pos : LT.lt 0 ε
    ⊢ Exists fun b => And (Membership.mem (↑(NormedAddCommGroup.toCompl.comp (Norm …
  -/
  rcases h.exists_pos with ⟨C', C'_pos, hC'⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    H : Type u_2
    inst✝ : SeminormedAddCommGroup H
    f : NormedAddGroupHom G H
    C : Real
    h : f.SurjectiveOnWith f.range C
    hatg : UniformSpace.Completion G
    hatg_in : Eq (f.completion hatg) 0
    ε : Real
    ε_pos : LT.lt 0 ε
    C' : Real
    C'_pos : GT.gt C' 0
    hC' : f.SurjectiveOnWith f.range C'
    ⊢ Exists fun b => And (Membership.mem (↑(NormedAddCommGroup.toCompl.comp (Norm …
  -/
  rcases exists_pos_mul_lt ε_pos (1 + C' * ‖f‖) with ⟨δ, δ_pos, hδ⟩
  obtain ⟨_, ⟨g : G, rfl⟩, hg : ‖hatg - g‖ < δ⟩ :=
    SeminormedAddCommGroup.mem_closure_iff.mp (Completion.isDenseInducing_coe.dense hatg) δ δ_pos
  /-
    case intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    H : Type u_2
    inst✝ : SeminormedAddCommGroup H
    f : NormedAddGroupHom G H
    C : Real
    h : f.SurjectiveOnWith f.range C
    hatg : UniformSpace.Completion G
    hatg_in : Eq (f.completion hatg) 0
    ε : Real
    ε_pos : LT.lt 0 ε
    C' : Real
    C'_pos : GT.gt C' 0
    hC' : f.SurjectiveOnWith f.range C'
    δ : Real
    δ_pos : LT.lt 0 δ
    hδ : LT.lt (HMul.hMul (HAdd.hAdd 1 (HMul.hMul C' (Norm.norm f))) δ) ε
    g : G
    hg : LT.lt (Norm.norm (HSub.hSub hatg (↑G g))) δ
    ⊢ Exists fun b => And (Membership.mem (↑(NormedAddCommGroup.toCompl.comp (Norm …
  -/
  obtain ⟨g' : G, hgg' : f g' = f g, hfg : ‖g'‖ ≤ C' * ‖f g‖⟩ := hC' (f g) (mem_range_self _ g)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    H : Type u_2
    inst✝ : SeminormedAddCommGroup H
    f : NormedAddGroupHom G H
    C : Real
    h : f.SurjectiveOnWith f.range C
    hatg : UniformSpace.Completion G
    hatg_in : Eq (f.completion hatg) 0
    ε : Real
    ε_pos : LT.lt 0 ε
    C' : Real
    C'_pos : GT.gt C' 0
    hC' : f.SurjectiveOnWith f.range C'
    δ : Real
    δ_pos : LT.lt 0 δ
    hδ : LT.lt (HMul.hMul (HAdd.hAdd 1 (HMul.hMul C' (Norm.norm f))) δ) ε
    g : G
    hg : LT.lt (Norm.norm (HSub.hSub hatg (↑G g))) δ
    g' : G
    hgg' : Eq (f g') (f g)
    hfg : LE.le (Norm.norm g') (HMul.hMul C' (Norm.norm (f g)))
    ⊢ Exists fun b => And (Membership.mem (↑(NormedAddCommGroup.toCompl.comp (Norm …
  -/
  have mem_ker : g - g' ∈ f.ker := by rw [f.mem_ker, map_sub, sub_eq_zero.mpr hgg'.symm]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝¹ : SeminormedAddCommGroup G
    H : Type u_2
    inst✝ : SeminormedAddCommGroup H
    f : NormedAddGroupHom G H
    C : Real
    h : f.SurjectiveOnWith f.range C
    hatg : UniformSpace.Completion G
    hatg_in : Eq (f.completion hatg) 0
    ε : Real
    ε_pos : LT.lt 0 ε
    C' : Real
    C'_pos : GT.gt C' 0
    hC' : f.SurjectiveOnWith f.range C'
    δ : Real
    δ_pos : LT.lt 0 δ
    hδ : LT.lt (HMul.hMul (HAdd.hAdd 1 (HMul.hMul C' (Norm.norm f))) δ) ε
    g : G
    hg : LT.lt (Norm.norm (HSub.hSub hatg (↑G g))) δ
    g' : G
    hgg' : Eq (f g') (f g)
    hfg : LE.le (Norm.norm g') (HMul.hMul C' (Norm.norm (f g)))
    mem_ker : Membership.mem f.ker (HSub.hSub g g')
    ⊢ Exists fun b => And (Membership.mem (↑(NormedAddCommGroup.toCompl.comp (Norm …
  -/
  refine ⟨_, ⟨⟨g - g', mem_ker⟩, rfl⟩, ?_⟩
  have : ‖f g‖ ≤ ‖f‖ * δ := calc
    ‖f g‖ ≤ ‖f‖ * ‖hatg - g‖ := by
      simpa [map_sub, hatg_in] using f.completion.le_opNorm (hatg - g)
    _ ≤ ‖f‖ * δ := by gcongr
  calc ‖hatg - ↑(g - g')‖ = ‖hatg - g + g'‖ := by rw [Completion.coe_sub, sub_add]
    _ ≤ ‖hatg - g‖ + ‖(g' : Completion G)‖ := norm_add_le _ _
    _ = ‖hatg - g‖ + ‖g'‖ := by rw [Completion.norm_coe]
    _ < δ + C' * ‖f g‖ := add_lt_add_of_lt_of_le hg hfg
    _ ≤ δ + C' * (‖f‖ * δ) := by gcongr
    _ < ε := by simpa only [add_mul, one_mul, mul_assoc] using hδ


/-- If `H` is complete, the extension of `f : NormedAddGroupHom G H` to a
`NormedAddGroupHom (completion G) H`. -/
def NormedAddGroupHom.extension (f : NormedAddGroupHom G H) : NormedAddGroupHom (Completion G) H :=
  .ofLipschitz (f.toAddMonoidHom.extension f.continuous) <|
    let _ := MetricSpace.ofT0PseudoMetricSpace H
    f.lipschitz.completion_extension


theorem NormedAddGroupHom.extension_def (f : NormedAddGroupHom G H) (v : G) :
    f.extension v = Completion.extension f v :=
  rfl


@[simp]
theorem NormedAddGroupHom.extension_coe (f : NormedAddGroupHom G H) (v : G) : f.extension v = f v :=
  AddMonoidHom.extension_coe _ f.continuous _


theorem NormedAddGroupHom.extension_coe_to_fun (f : NormedAddGroupHom G H) :
    (f.extension : Completion G → H) = Completion.extension f :=
  rfl


theorem NormedAddGroupHom.extension_unique (f : NormedAddGroupHom G H)
    {g : NormedAddGroupHom (Completion G) H} (hg : ∀ v, f v = g v) : f.extension = g := by
  /-
    G : Type u_1
    inst✝³ : SeminormedAddCommGroup G
    H : Type u_2
    inst✝² : SeminormedAddCommGroup H
    inst✝¹ : T0Space H
    inst✝ : CompleteSpace H
    f : NormedAddGroupHom G H
    g : NormedAddGroupHom (UniformSpace.Completion G) H
    hg : ∀ (v : G), Eq (f v) (g (↑G v))
    ⊢ Eq f.extension g
  -/
  ext v
  rw [NormedAddGroupHom.extension_coe_to_fun,
    Completion.extension_unique f.uniformContinuous g.uniformContinuous fun a => hg a]


