local notation "hat" => Completion


/-- A topological field is completable if it is separated and the image under
the mapping x ↦ x⁻¹ of every Cauchy filter (with respect to the additive uniform structure)
which does not have a cluster point at 0 is a Cauchy filter
(with respect to the additive uniform structure). This ensures the completion is
a field.
-/
class CompletableTopField extends T0Space K : Prop where
  nice : ∀ F : Filter K, Cauchy F → 𝓝 0 ⊓ F = ⊥ → Cauchy (map (fun x => x⁻¹) F)


instance (priority := 100) [T0Space K] : Nontrivial (hat K) :=
  ⟨⟨0, 1, fun h => zero_ne_one <| (isUniformEmbedding_coe K).injective h⟩⟩


/-- extension of inversion to the completion of a field. -/
def hatInv : hat K → hat K :=
  isDenseInducing_coe.extend fun x : K => (↑x⁻¹ : hat K)


theorem continuous_hatInv [CompletableTopField K] {x : hat K} (h : x ≠ 0) :
    ContinuousAt hatInv x := by
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : UniformSpace K
    inst✝ : CompletableTopField K
    x : UniformSpace.Completion K
    h : Ne x 0
    ⊢ ContinuousAt UniformSpace.Completion.hatInv x
  -/
  refine isDenseInducing_coe.continuousAt_extend ?_
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : UniformSpace K
    inst✝ : CompletableTopField K
    x : UniformSpace.Completion K
    h : Ne x 0
    ⊢ Filter.Eventually (fun x => Exists fun c => Filter.Tendsto (fun x => ↑K (Inv …
  -/
  apply mem_of_superset (compl_singleton_mem_nhds h)
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : UniformSpace K
    inst✝ : CompletableTopField K
    x : UniformSpace.Completion K
    h : Ne x 0
    ⊢ HasSubset.Subset (HasCompl.compl (Singleton.singleton 0)) (setOf fun x => (f …
  -/
  intro y y_ne
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : UniformSpace K
    inst✝ : CompletableTopField K
    x : UniformSpace.Completion K
    h : Ne x 0
    y : UniformSpace.Completion K
    y_ne : Membership.mem (HasCompl.compl (Singleton.singleton 0)) y
    ⊢ Membership.mem (setOf fun x => (fun x => Exists fun c => Filter.Tendsto (fun …
  -/
  rw [mem_compl_singleton_iff] at y_ne
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : UniformSpace K
    inst✝ : CompletableTopField K
    x : UniformSpace.Completion K
    h : Ne x 0
    y : UniformSpace.Completion K
    y_ne : Ne y 0
    ⊢ Membership.mem (setOf fun x => (fun x => Exists fun c => Filter.Tendsto (fun …
  -/
  apply CompleteSpace.complete
  have : (fun (x : K) => (↑x⁻¹ : hat K)) =
      ((fun (y : K) => (↑y : hat K))∘(fun (x : K) => (x⁻¹ : K))) := by
    unfold Function.comp
    simp
  /-
    case a
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : UniformSpace K
    inst✝ : CompletableTopField K
    x : UniformSpace.Completion K
    h : Ne x 0
    y : UniformSpace.Completion K
    y_ne : Ne y 0
    this : Eq (fun x => ↑K (Inv.inv x)) (Function.comp (fun y => ↑K y) fun x => In …
    ⊢ Cauchy (Filter.map (fun x => ↑K (Inv.inv x)) (Filter.comap (↑K) (nhds y)))
  -/
  rw [this, ← Filter.map_map]
  /-
    case a
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : UniformSpace K
    inst✝ : CompletableTopField K
    x : UniformSpace.Completion K
    h : Ne x 0
    y : UniformSpace.Completion K
    y_ne : Ne y 0
    this : Eq (fun x => ↑K (Inv.inv x)) (Function.comp (fun y => ↑K y) fun x => In …
    ⊢ Cauchy (Filter.map (fun y => ↑K y) (Filter.map (fun x => Inv.inv x) (Filter. …
  -/
  apply Cauchy.map _ (Completion.uniformContinuous_coe K)
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : UniformSpace K
    inst✝ : CompletableTopField K
    x : UniformSpace.Completion K
    h : Ne x 0
    y : UniformSpace.Completion K
    y_ne : Ne y 0
    this : Eq (fun x => ↑K (Inv.inv x)) (Function.comp (fun y => ↑K y) fun x => In …
    ⊢ Cauchy (Filter.map (fun x => Inv.inv x) (Filter.comap (↑K) (nhds y)))
  -/
  apply CompletableTopField.nice
    /-
      case a
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : UniformSpace K
      inst✝ : CompletableTopField K
      x : UniformSpace.Completion K
      h : Ne x 0
      y : UniformSpace.Completion K
      y_ne : Ne y 0
      this : Eq (fun x => ↑K (Inv.inv x)) (Function.comp (fun y => ↑K y) fun x => In …
      ⊢ Cauchy (Filter.comap (↑K) (nhds y))
    -/
  · haveI := isDenseInducing_coe.comap_nhds_neBot y
    /-
      case a
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : UniformSpace K
      inst✝ : CompletableTopField K
      x : UniformSpace.Completion K
      h : Ne x 0
      y : UniformSpace.Completion K
      y_ne : Ne y 0
      this✝ : Eq (fun x => ↑K (Inv.inv x)) (Function.comp (fun y => ↑K y) fun x => I …
      this : (Filter.comap (↑K) (nhds y)).NeBot
      ⊢ Cauchy (Filter.comap (↑K) (nhds y))
    -/
    apply cauchy_nhds.comap
    /-
      case a.hm
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : UniformSpace K
      inst✝ : CompletableTopField K
      x : UniformSpace.Completion K
      h : Ne x 0
      y : UniformSpace.Completion K
      y_ne : Ne y 0
      this✝ : Eq (fun x => ↑K (Inv.inv x)) (Function.comp (fun y => ↑K y) fun x => I …
      this : (Filter.comap (↑K) (nhds y)).NeBot
      ⊢ LE.le (Filter.comap (fun p => { fst := ↑K p.1, snd := ↑K p.2 }) (uniformity  …
    -/
    rw [Completion.comap_coe_eq_uniformity]
    /-
      🎉 no goals
    -/
  · have eq_bot : 𝓝 (0 : hat K) ⊓ 𝓝 y = ⊥ := by
      by_contra h
      exact y_ne (eq_of_nhds_neBot <| neBot_iff.mpr h).symm
    /-
      case a
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : UniformSpace K
      inst✝ : CompletableTopField K
      x : UniformSpace.Completion K
      h : Ne x 0
      y : UniformSpace.Completion K
      y_ne : Ne y 0
      this : Eq (fun x => ↑K (Inv.inv x)) (Function.comp (fun y => ↑K y) fun x => In …
      eq_bot : Eq (Min.min (nhds 0) (nhds y)) Bot.bot
      ⊢ Eq (Min.min (nhds 0) (Filter.comap (↑K) (nhds y))) Bot.bot
    -/
    erw [isDenseInducing_coe.nhds_eq_comap (0 : K), ← Filter.comap_inf, eq_bot]
    /-
      case a
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : UniformSpace K
      inst✝ : CompletableTopField K
      x : UniformSpace.Completion K
      h : Ne x 0
      y : UniformSpace.Completion K
      y_ne : Ne y 0
      this : Eq (fun x => ↑K (Inv.inv x)) (Function.comp (fun y => ↑K y) fun x => In …
      eq_bot : Eq (Min.min (nhds 0) (nhds y)) Bot.bot
      ⊢ Eq (Filter.comap (↑K) Bot.bot) Bot.bot
    -/
    exact comap_bot
    /-
      🎉 no goals
    -/


open Classical in
/-
The value of `hat_inv` at zero is not really specified, although it's probably zero.
Here we explicitly enforce the `inv_zero` axiom.
-/
instance instInvCompletion : Inv (hat K) :=
  ⟨fun x => if x = 0 then 0 else hatInv x⟩


theorem hatInv_extends {x : K} (h : x ≠ 0) : hatInv (x : hat K) = ↑(x⁻¹ : K) :=
  isDenseInducing_coe.extend_eq_at ((continuous_coe K).continuousAt.comp (continuousAt_inv₀ h))


@[norm_cast]
theorem coe_inv (x : K) : (x : hat K)⁻¹ = ((x⁻¹ : K) : hat K) := by
  /-
    K : Type u_1
    inst✝³ : Field K
    inst✝² : UniformSpace K
    inst✝¹ : TopologicalDivisionRing K
    inst✝ : CompletableTopField K
    x : K
    ⊢ Eq (Inv.inv (↑K x)) (↑K (Inv.inv x))
  -/
  by_cases h : x = 0
    /-
      case pos
      K : Type u_1
      inst✝³ : Field K
      inst✝² : UniformSpace K
      inst✝¹ : TopologicalDivisionRing K
      inst✝ : CompletableTopField K
      x : K
      h : Eq x 0
      ⊢ Eq (Inv.inv (↑K x)) (↑K (Inv.inv x))
    -/
  · rw [h, inv_zero]
    /-
      case pos
      K : Type u_1
      inst✝³ : Field K
      inst✝² : UniformSpace K
      inst✝¹ : TopologicalDivisionRing K
      inst✝ : CompletableTopField K
      x : K
      h : Eq x 0
      ⊢ Eq (Inv.inv (↑K 0)) (↑K 0)
    -/
    dsimp [Inv.inv]
    /-
      case pos
      K : Type u_1
      inst✝³ : Field K
      inst✝² : UniformSpace K
      inst✝¹ : TopologicalDivisionRing K
      inst✝ : CompletableTopField K
      x : K
      h : Eq x 0
      ⊢ Eq (ite (Eq (↑K 0) 0) 0 (↑K 0).hatInv) (↑K 0)
    -/
    norm_cast
    /-
      case pos
      K : Type u_1
      inst✝³ : Field K
      inst✝² : UniformSpace K
      inst✝¹ : TopologicalDivisionRing K
      inst✝ : CompletableTopField K
      x : K
      h : Eq x 0
      ⊢ Eq (ite (Eq 0 0) 0 (UniformSpace.Completion.hatInv 0)) 0
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u_1
      inst✝³ : Field K
      inst✝² : UniformSpace K
      inst✝¹ : TopologicalDivisionRing K
      inst✝ : CompletableTopField K
      x : K
      h : Not (Eq x 0)
      ⊢ Eq (Inv.inv (↑K x)) (↑K (Inv.inv x))
    -/
  · conv_lhs => dsimp [Inv.inv]
    /-
      case neg
      K : Type u_1
      inst✝³ : Field K
      inst✝² : UniformSpace K
      inst✝¹ : TopologicalDivisionRing K
      inst✝ : CompletableTopField K
      x : K
      h : Not (Eq x 0)
      ⊢ Eq (ite (Eq (↑K x) 0) 0 (↑K x).hatInv) (↑K (Inv.inv x))
    -/
    rw [if_neg]
      /-
        case neg
        K : Type u_1
        inst✝³ : Field K
        inst✝² : UniformSpace K
        inst✝¹ : TopologicalDivisionRing K
        inst✝ : CompletableTopField K
        x : K
        h : Not (Eq x 0)
        ⊢ Eq (↑K x).hatInv (↑K (Inv.inv x))
      -/
    · exact hatInv_extends h
      /-
        🎉 no goals
      -/
      /-
        case neg.hnc
        K : Type u_1
        inst✝³ : Field K
        inst✝² : UniformSpace K
        inst✝¹ : TopologicalDivisionRing K
        inst✝ : CompletableTopField K
        x : K
        h : Not (Eq x 0)
        ⊢ Not (Eq (↑K x) 0)
      -/
    · exact fun H => h (isDenseEmbedding_coe.injective H)
      /-
        🎉 no goals
      -/


theorem mul_hatInv_cancel {x : hat K} (x_ne : x ≠ 0) : x * hatInv x = 1 := by
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : UniformSpace K
    inst✝² : TopologicalDivisionRing K
    inst✝¹ : CompletableTopField K
    inst✝ : UniformAddGroup K
    x : UniformSpace.Completion K
    x_ne : Ne x 0
    ⊢ Eq (HMul.hMul x x.hatInv) 1
  -/
  haveI : T1Space (hat K) := T2Space.t1Space
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : UniformSpace K
    inst✝² : TopologicalDivisionRing K
    inst✝¹ : CompletableTopField K
    inst✝ : UniformAddGroup K
    x : UniformSpace.Completion K
    x_ne : Ne x 0
    this : T1Space (UniformSpace.Completion K)
    ⊢ Eq (HMul.hMul x x.hatInv) 1
  -/
  let f := fun x : hat K => x * hatInv x
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : UniformSpace K
    inst✝² : TopologicalDivisionRing K
    inst✝¹ : CompletableTopField K
    inst✝ : UniformAddGroup K
    x : UniformSpace.Completion K
    x_ne : Ne x 0
    this : T1Space (UniformSpace.Completion K)
    f : UniformSpace.Completion K → UniformSpace.Completion K := fun x => HMul.hMu …
    ⊢ Eq (HMul.hMul x x.hatInv) 1
  -/
  let c := (fun (x : K) => (x : hat K))
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : UniformSpace K
    inst✝² : TopologicalDivisionRing K
    inst✝¹ : CompletableTopField K
    inst✝ : UniformAddGroup K
    x : UniformSpace.Completion K
    x_ne : Ne x 0
    this : T1Space (UniformSpace.Completion K)
    f : UniformSpace.Completion K → UniformSpace.Completion K := fun x => HMul.hMu …
    c : K → UniformSpace.Completion K := fun x => ↑K x
    ⊢ Eq (HMul.hMul x x.hatInv) 1
  -/
  change f x = 1
  have cont : ContinuousAt f x := by
    letI : TopologicalSpace (hat K × hat K) := instTopologicalSpaceProd
    have : ContinuousAt (fun y : hat K => ((y, hatInv y) : hat K × hat K)) x :=
      continuous_id.continuousAt.prod (continuous_hatInv x_ne)
    exact (_root_.continuous_mul.continuousAt.comp this : _)
  have clo : x ∈ closure (c '' {0}ᶜ) := by
    have := isDenseInducing_coe.dense x
    rw [← image_univ, show (univ : Set K) = {0} ∪ {0}ᶜ from (union_compl_self _).symm,
      image_union] at this
    apply mem_closure_of_mem_closure_union this
    rw [image_singleton]
    exact compl_singleton_mem_nhds x_ne
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : UniformSpace K
    inst✝² : TopologicalDivisionRing K
    inst✝¹ : CompletableTopField K
    inst✝ : UniformAddGroup K
    x : UniformSpace.Completion K
    x_ne : Ne x 0
    this : T1Space (UniformSpace.Completion K)
    f : UniformSpace.Completion K → UniformSpace.Completion K := fun x => HMul.hMu …
    c : K → UniformSpace.Completion K := fun x => ↑K x
    cont : ContinuousAt f x
    clo : Membership.mem (closure (Set.image c (HasCompl.compl (Singleton.singleto …
    ⊢ Eq (f x) 1
  -/
  have fxclo : f x ∈ closure (f '' (c '' {0}ᶜ)) := mem_closure_image cont clo
  have : f '' (c '' {0}ᶜ) ⊆ {1} := by
    rw [image_image]
    rintro _ ⟨z, z_ne, rfl⟩
    rw [mem_singleton_iff]
    rw [mem_compl_singleton_iff] at z_ne
    dsimp [f]
    rw [hatInv_extends z_ne, ← coe_mul]
    rw [mul_inv_cancel₀ z_ne, coe_one]
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : UniformSpace K
    inst✝² : TopologicalDivisionRing K
    inst✝¹ : CompletableTopField K
    inst✝ : UniformAddGroup K
    x : UniformSpace.Completion K
    x_ne : Ne x 0
    this✝ : T1Space (UniformSpace.Completion K)
    f : UniformSpace.Completion K → UniformSpace.Completion K := fun x => HMul.hMu …
    c : K → UniformSpace.Completion K := fun x => ↑K x
    cont : ContinuousAt f x
    clo : Membership.mem (closure (Set.image c (HasCompl.compl (Singleton.singleto …
    fxclo : Membership.mem (closure (Set.image f (Set.image c (HasCompl.compl (Sin …
    this : HasSubset.Subset (Set.image f (Set.image c (HasCompl.compl (Singleton.s …
    ⊢ Eq (f x) 1
  -/
  replace fxclo := closure_mono this fxclo
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : UniformSpace K
    inst✝² : TopologicalDivisionRing K
    inst✝¹ : CompletableTopField K
    inst✝ : UniformAddGroup K
    x : UniformSpace.Completion K
    x_ne : Ne x 0
    this✝ : T1Space (UniformSpace.Completion K)
    f : UniformSpace.Completion K → UniformSpace.Completion K := fun x => HMul.hMu …
    c : K → UniformSpace.Completion K := fun x => ↑K x
    cont : ContinuousAt f x
    clo : Membership.mem (closure (Set.image c (HasCompl.compl (Singleton.singleto …
    this : HasSubset.Subset (Set.image f (Set.image c (HasCompl.compl (Singleton.s …
    fxclo : Membership.mem (closure (Singleton.singleton 1)) (f x)
    ⊢ Eq (f x) 1
  -/
  rwa [closure_singleton, mem_singleton_iff] at fxclo
  /-
    🎉 no goals
  -/


instance instField : Field (hat K) where
  exists_pair_ne := ⟨0, 1, fun h => zero_ne_one ((isUniformEmbedding_coe K).injective h)⟩
                                     /-
                                       K : Type u_1
                                       inst✝⁴ : Field K
                                       inst✝³ : UniformSpace K
                                       inst✝² : TopologicalDivisionRing K
                                       inst✝¹ : CompletableTopField K
                                       inst✝ : UniformAddGroup K
                                       x : UniformSpace.Completion K
                                       x_ne : Ne x 0
                                       ⊢ Eq (HMul.hMul x (Inv.inv x)) 1
                                     -/
  mul_inv_cancel := fun x x_ne => by simp only [Inv.inv, if_neg x_ne, mul_hatInv_cancel x_ne]
                                     /-
                                       🎉 no goals
                                     -/
                 /-
                   K : Type u_1
                   inst✝⁴ : Field K
                   inst✝³ : UniformSpace K
                   inst✝² : TopologicalDivisionRing K
                   inst✝¹ : CompletableTopField K
                   inst✝ : UniformAddGroup K
                   ⊢ Eq (Inv.inv 0) 0
                 -/
  inv_zero := by simp only [Inv.inv, ite_true]
                 /-
                   🎉 no goals
                 -/
  -- TODO: use a better defeq
  nnqsmul := _
  nnqsmul_def := fun _ _ => rfl
  qsmul := _
  qsmul_def := fun _ _ => rfl


instance : TopologicalDivisionRing (hat K) :=
  { Completion.topologicalRing with
    continuousAt_inv₀ := by
      /-
        K : Type u_1
        inst✝⁴ : Field K
        inst✝³ : UniformSpace K
        inst✝² : TopologicalDivisionRing K
        inst✝¹ : CompletableTopField K
        inst✝ : UniformAddGroup K
        ⊢ ∀ ⦃x : UniformSpace.Completion K⦄, Ne x 0 → ContinuousAt Inv.inv x
      -/
      intro x x_ne
      have : { y | hatInv y = y⁻¹ } ∈ 𝓝 x :=
        haveI : {(0 : hat K)}ᶜ ⊆ { y : hat K | hatInv y = y⁻¹ } := by
          intro y y_ne
          rw [mem_compl_singleton_iff] at y_ne
          dsimp [Inv.inv]
          rw [if_neg y_ne]
        mem_of_superset (compl_singleton_mem_nhds x_ne) this
      /-
        K : Type u_1
        inst✝⁴ : Field K
        inst✝³ : UniformSpace K
        inst✝² : TopologicalDivisionRing K
        inst✝¹ : CompletableTopField K
        inst✝ : UniformAddGroup K
        x : UniformSpace.Completion K
        x_ne : Ne x 0
        this : Membership.mem (nhds x) (setOf fun y => Eq y.hatInv (Inv.inv y))
        ⊢ ContinuousAt Inv.inv x
      -/
      exact ContinuousAt.congr (continuous_hatInv x_ne) this }
      /-
        🎉 no goals
      -/


instance Subfield.completableTopField (K : Subfield L) : CompletableTopField K where
  nice F F_cau inf_F := by
    /-
      K✝ : Type u_1
      inst✝⁴ : Field K✝
      inst✝³ : UniformSpace K✝
      L : Type u_2
      inst✝² : Field L
      inst✝¹ : UniformSpace L
      inst✝ : CompletableTopField L
      K : Subfield L
      F : Filter (Subtype fun x => Membership.mem K x)
      F_cau : Cauchy F
      inf_F : Eq (Min.min (nhds 0) F) Bot.bot
      ⊢ Cauchy (Filter.map (fun x => Inv.inv x) F)
    -/
    let i : K →+* L := K.subtype
    /-
      K✝ : Type u_1
      inst✝⁴ : Field K✝
      inst✝³ : UniformSpace K✝
      L : Type u_2
      inst✝² : Field L
      inst✝¹ : UniformSpace L
      inst✝ : CompletableTopField L
      K : Subfield L
      F : Filter (Subtype fun x => Membership.mem K x)
      F_cau : Cauchy F
      inf_F : Eq (Min.min (nhds 0) F) Bot.bot
      i : RingHom (Subtype fun x => Membership.mem K x) L := K.subtype
      ⊢ Cauchy (Filter.map (fun x => Inv.inv x) F)
    -/
    have hi : IsUniformInducing i := isUniformEmbedding_subtype_val.isUniformInducing
    /-
      K✝ : Type u_1
      inst✝⁴ : Field K✝
      inst✝³ : UniformSpace K✝
      L : Type u_2
      inst✝² : Field L
      inst✝¹ : UniformSpace L
      inst✝ : CompletableTopField L
      K : Subfield L
      F : Filter (Subtype fun x => Membership.mem K x)
      F_cau : Cauchy F
      inf_F : Eq (Min.min (nhds 0) F) Bot.bot
      i : RingHom (Subtype fun x => Membership.mem K x) L := K.subtype
      hi : IsUniformInducing ⇑i
      ⊢ Cauchy (Filter.map (fun x => Inv.inv x) F)
    -/
    rw [← hi.cauchy_map_iff] at F_cau ⊢
    /-
      K✝ : Type u_1
      inst✝⁴ : Field K✝
      inst✝³ : UniformSpace K✝
      L : Type u_2
      inst✝² : Field L
      inst✝¹ : UniformSpace L
      inst✝ : CompletableTopField L
      K : Subfield L
      F : Filter (Subtype fun x => Membership.mem K x)
      inf_F : Eq (Min.min (nhds 0) F) Bot.bot
      i : RingHom (Subtype fun x => Membership.mem K x) L := K.subtype
      F_cau : Cauchy (Filter.map (⇑i) F)
      hi : IsUniformInducing ⇑i
      ⊢ Cauchy (Filter.map (⇑i) (Filter.map (fun x => Inv.inv x) F))
    -/
    rw [map_comm (show (i ∘ fun x => x⁻¹) = (fun x => x⁻¹) ∘ i by ext; rfl)]
    /-
      K✝ : Type u_1
      inst✝⁴ : Field K✝
      inst✝³ : UniformSpace K✝
      L : Type u_2
      inst✝² : Field L
      inst✝¹ : UniformSpace L
      inst✝ : CompletableTopField L
      K : Subfield L
      F : Filter (Subtype fun x => Membership.mem K x)
      inf_F : Eq (Min.min (nhds 0) F) Bot.bot
      i : RingHom (Subtype fun x => Membership.mem K x) L := K.subtype
      F_cau : Cauchy (Filter.map (⇑i) F)
      hi : IsUniformInducing ⇑i
      ⊢ Cauchy (Filter.map (fun x => Inv.inv x) (Filter.map (⇑i) F))
    -/
    apply CompletableTopField.nice _ F_cau
    /-
      K✝ : Type u_1
      inst✝⁴ : Field K✝
      inst✝³ : UniformSpace K✝
      L : Type u_2
      inst✝² : Field L
      inst✝¹ : UniformSpace L
      inst✝ : CompletableTopField L
      K : Subfield L
      F : Filter (Subtype fun x => Membership.mem K x)
      inf_F : Eq (Min.min (nhds 0) F) Bot.bot
      i : RingHom (Subtype fun x => Membership.mem K x) L := K.subtype
      F_cau : Cauchy (Filter.map (⇑i) F)
      hi : IsUniformInducing ⇑i
      ⊢ Eq (Min.min (nhds 0) (Filter.map (⇑i) F)) Bot.bot
    -/
    rw [← Filter.push_pull', ← map_zero i, ← hi.isInducing.nhds_eq_comap, inf_F, Filter.map_bot]
    /-
      🎉 no goals
    -/


instance (priority := 100) completableTopField_of_complete (L : Type*) [Field L] [UniformSpace L]
    [TopologicalDivisionRing L] [T0Space L] [CompleteSpace L] : CompletableTopField L where
  nice F cau_F hF := by
    /-
      K : Type u_1
      inst✝⁹ : Field K
      inst✝⁸ : UniformSpace K
      L✝ : Type u_2
      inst✝⁷ : Field L✝
      inst✝⁶ : UniformSpace L✝
      inst✝⁵ : CompletableTopField L✝
      L : Type u_3
      inst✝⁴ : Field L
      inst✝³ : UniformSpace L
      inst✝² : TopologicalDivisionRing L
      inst✝¹ : T0Space L
      inst✝ : CompleteSpace L
      F : Filter L
      cau_F : Cauchy F
      hF : Eq (Min.min (nhds 0) F) Bot.bot
      ⊢ Cauchy (Filter.map (fun x => Inv.inv x) F)
    -/
    haveI : NeBot F := cau_F.1
    /-
      K : Type u_1
      inst✝⁹ : Field K
      inst✝⁸ : UniformSpace K
      L✝ : Type u_2
      inst✝⁷ : Field L✝
      inst✝⁶ : UniformSpace L✝
      inst✝⁵ : CompletableTopField L✝
      L : Type u_3
      inst✝⁴ : Field L
      inst✝³ : UniformSpace L
      inst✝² : TopologicalDivisionRing L
      inst✝¹ : T0Space L
      inst✝ : CompleteSpace L
      F : Filter L
      cau_F : Cauchy F
      hF : Eq (Min.min (nhds 0) F) Bot.bot
      this : F.NeBot
      ⊢ Cauchy (Filter.map (fun x => Inv.inv x) F)
    -/
    rcases CompleteSpace.complete cau_F with ⟨x, hx⟩
    have hx' : x ≠ 0 := by
      rintro rfl
      rw [inf_eq_right.mpr hx] at hF
      exact cau_F.1.ne hF
    exact Filter.Tendsto.cauchy_map <|
      calc
        map (fun x => x⁻¹) F ≤ map (fun x => x⁻¹) (𝓝 x) := map_mono hx
        _ ≤ 𝓝 x⁻¹ := continuousAt_inv₀ hx'


/-- The pullback of a completable topological field along a uniform inducing
ring homomorphism is a completable topological field. -/
theorem IsUniformInducing.completableTopField
    [UniformSpace α] [T0Space α]
    {f : α →+* β} (hf : IsUniformInducing f) :
    CompletableTopField α := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝⁴ : Field β
    b : UniformSpace β
    inst✝³ : CompletableTopField β
    inst✝² : Field α
    inst✝¹ : UniformSpace α
    inst✝ : T0Space α
    f : RingHom α β
    hf : IsUniformInducing ⇑f
    ⊢ CompletableTopField α
  -/
  refine CompletableTopField.mk (fun F F_cau inf_F => ?_)
  /-
    α : Type u_3
    β : Type u_4
    inst✝⁴ : Field β
    b : UniformSpace β
    inst✝³ : CompletableTopField β
    inst✝² : Field α
    inst✝¹ : UniformSpace α
    inst✝ : T0Space α
    f : RingHom α β
    hf : IsUniformInducing ⇑f
    F : Filter α
    F_cau : Cauchy F
    inf_F : Eq (Min.min (nhds 0) F) Bot.bot
    ⊢ Cauchy (Filter.map (fun x => Inv.inv x) F)
  -/
  rw [← IsUniformInducing.cauchy_map_iff hf] at F_cau ⊢
  have h_comm : (f ∘ fun x => x⁻¹) = (fun x => x⁻¹) ∘ f := by
    ext; simp only [Function.comp_apply, map_inv₀, Subfield.coe_inv]
  /-
    α : Type u_3
    β : Type u_4
    inst✝⁴ : Field β
    b : UniformSpace β
    inst✝³ : CompletableTopField β
    inst✝² : Field α
    inst✝¹ : UniformSpace α
    inst✝ : T0Space α
    f : RingHom α β
    hf : IsUniformInducing ⇑f
    F : Filter α
    F_cau : Cauchy (Filter.map (⇑f) F)
    inf_F : Eq (Min.min (nhds 0) F) Bot.bot
    h_comm : Eq (Function.comp ⇑f fun x => Inv.inv x) (Function.comp (fun x => Inv …
    ⊢ Cauchy (Filter.map (⇑f) (Filter.map (fun x => Inv.inv x) F))
  -/
  rw [Filter.map_comm h_comm]
  /-
    α : Type u_3
    β : Type u_4
    inst✝⁴ : Field β
    b : UniformSpace β
    inst✝³ : CompletableTopField β
    inst✝² : Field α
    inst✝¹ : UniformSpace α
    inst✝ : T0Space α
    f : RingHom α β
    hf : IsUniformInducing ⇑f
    F : Filter α
    F_cau : Cauchy (Filter.map (⇑f) F)
    inf_F : Eq (Min.min (nhds 0) F) Bot.bot
    h_comm : Eq (Function.comp ⇑f fun x => Inv.inv x) (Function.comp (fun x => Inv …
    ⊢ Cauchy (Filter.map (fun x => Inv.inv x) (Filter.map (⇑f) F))
  -/
  apply CompletableTopField.nice _ F_cau
  /-
    α : Type u_3
    β : Type u_4
    inst✝⁴ : Field β
    b : UniformSpace β
    inst✝³ : CompletableTopField β
    inst✝² : Field α
    inst✝¹ : UniformSpace α
    inst✝ : T0Space α
    f : RingHom α β
    hf : IsUniformInducing ⇑f
    F : Filter α
    F_cau : Cauchy (Filter.map (⇑f) F)
    inf_F : Eq (Min.min (nhds 0) F) Bot.bot
    h_comm : Eq (Function.comp ⇑f fun x => Inv.inv x) (Function.comp (fun x => Inv …
    ⊢ Eq (Min.min (nhds 0) (Filter.map (⇑f) F)) Bot.bot
  -/
  rw [← Filter.push_pull', ← map_zero f, ← hf.isInducing.nhds_eq_comap, inf_F, Filter.map_bot]
  /-
    🎉 no goals
  -/

