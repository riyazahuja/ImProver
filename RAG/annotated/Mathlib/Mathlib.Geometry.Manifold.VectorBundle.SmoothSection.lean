local notation "∞" => (⊤ : ℕ∞)


/-- Bundled `n` times continuously differentiable sections of a vector bundle. -/
structure ContMDiffSection where
  /-- the underlying function of this section -/
  protected toFun : ∀ x, V x
  /-- proof that this section is `C^n` -/
  protected contMDiff_toFun : ContMDiff I (I.prod 𝓘(𝕜, F)) n fun x ↦
    TotalSpace.mk' F x (toFun x)


@[deprecated (since := "024-11-21")] alias SmoothSection := ContMDiffSection


@[inherit_doc] scoped[Manifold] notation "Cₛ^" n "⟮" I "; " F ", " V "⟯" => ContMDiffSection I F n V


instance : DFunLike Cₛ^n⟮I; F, V⟯ M V where
  coe := ContMDiffSection.toFun
                       /-
                         𝕜 : Type u_1
                         inst✝¹⁰ : NontriviallyNormedField 𝕜
                         E : Type u_2
                         inst✝⁹ : NormedAddCommGroup E
                         inst✝⁸ : NormedSpace 𝕜 E
                         H : Type u_3
                         inst✝⁷ : TopologicalSpace H
                         I : ModelWithCorners 𝕜 E H
                         M : Type u_4
                         inst✝⁶ : TopologicalSpace M
                         inst✝⁵ : ChartedSpace H M
                         F : Type u_5
                         inst✝⁴ : NormedAddCommGroup F
                         inst✝³ : NormedSpace 𝕜 F
                         n : ENat
                         V : M → Type u_6
                         inst✝² : TopologicalSpace (Bundle.TotalSpace F V)
                         inst✝¹ : (x : M) → TopologicalSpace (V x)
                         inst✝ : FiberBundle F V
                         ⊢ Function.Injective ContMDiffSection.toFun
                       -/
  coe_injective' := by rintro ⟨⟩ ⟨⟩ h; congr
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem coeFn_mk (s : ∀ x, V x)
    (hs : ContMDiff I (I.prod 𝓘(𝕜, F)) n fun x => TotalSpace.mk x (s x)) :
    (mk s hs : ∀ x, V x) = s :=
  rfl


protected theorem contMDiff (s : Cₛ^n⟮I; F, V⟯) :
    ContMDiff I (I.prod 𝓘(𝕜, F)) n fun x => TotalSpace.mk' F x (s x : V x) :=
  s.contMDiff_toFun


@[deprecated (since := "2024-11-21")] alias smooth := ContMDiffSection.contMDiff


theorem coe_inj ⦃s t : Cₛ^n⟮I; F, V⟯⦄ (h : (s : ∀ x, V x) = t) : s = t :=
  DFunLike.ext' h


theorem coe_injective : Injective ((↑) : Cₛ^n⟮I; F, V⟯ → ∀ x, V x) :=
  coe_inj


@[ext]
theorem ext (h : ∀ x, s x = t x) : s = t := DFunLike.ext _ _ h


instance instAdd : Add Cₛ^n⟮I; F, V⟯ := by
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    ⊢ Add (ContMDiffSection I F n V)
  -/
  refine ⟨fun s t => ⟨s + t, ?_⟩⟩
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    ⊢ ContMDiff I (I.prod (modelWithCornersSelf 𝕜 F)) n fun x => Bundle.TotalSpace …
  -/
  intro x₀
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    x₀ : M
    ⊢ ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.TotalSp …
  -/
  have hs := s.contMDiff x₀
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    x₀ : M
    hs : ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.Tota …
    ⊢ ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.TotalSp …
  -/
  have ht := t.contMDiff x₀
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    x₀ : M
    hs : ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.Tota …
    ht : ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.Tota …
    ⊢ ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.TotalSp …
  -/
  rw [contMDiffAt_section] at hs ht ⊢
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    x₀ : M
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑(FiberBundle.trivi …
    ht : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑(FiberBundle.trivi …
    ⊢ ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑(FiberBundle.triviali …
  -/
  set e := trivializationAt F V x₀
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    x₀ : M
    e : Trivialization F Bundle.TotalSpace.proj := FiberBundle.trivializationAt F  …
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ht : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ⊢ ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd := …
  -/
  refine (hs.add ht).congr_of_eventuallyEq ?_
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    x₀ : M
    e : Trivialization F Bundle.TotalSpace.proj := FiberBundle.trivializationAt F  …
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ht : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ⊢ (nhds x₀).EventuallyEq (fun x => (↑e { proj := x, snd := HAdd.hAdd (⇑s) (⇑t) …
  -/
  refine eventually_of_mem (e.open_baseSet.mem_nhds <| mem_baseSet_trivializationAt F V x₀) ?_
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    x₀ : M
    e : Trivialization F Bundle.TotalSpace.proj := FiberBundle.trivializationAt F  …
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ht : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ⊢ ∀ (x : M), Membership.mem e.baseSet x → Eq ((fun x => (↑e { proj := x, snd : …
  -/
  intro x hx
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    x₀ : M
    e : Trivialization F Bundle.TotalSpace.proj := FiberBundle.trivializationAt F  …
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ht : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    x : M
    hx : Membership.mem e.baseSet x
    ⊢ Eq ((fun x => (↑e { proj := x, snd := HAdd.hAdd (⇑s) (⇑t) x }).2) x) (HAdd.h …
  -/
  apply (e.linear 𝕜 hx).1
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_add (s t : Cₛ^n⟮I; F, V⟯) : ⇑(s + t) = ⇑s + t :=
  rfl


instance instSub : Sub Cₛ^n⟮I; F, V⟯ := by
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    ⊢ Sub (ContMDiffSection I F n V)
  -/
  refine ⟨fun s t => ⟨s - t, ?_⟩⟩
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    ⊢ ContMDiff I (I.prod (modelWithCornersSelf 𝕜 F)) n fun x => Bundle.TotalSpace …
  -/
  intro x₀
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    x₀ : M
    ⊢ ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.TotalSp …
  -/
  have hs := s.contMDiff x₀
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    x₀ : M
    hs : ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.Tota …
    ⊢ ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.TotalSp …
  -/
  have ht := t.contMDiff x₀
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    x₀ : M
    hs : ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.Tota …
    ht : ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.Tota …
    ⊢ ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.TotalSp …
  -/
  rw [contMDiffAt_section] at hs ht ⊢
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    x₀ : M
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑(FiberBundle.trivi …
    ht : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑(FiberBundle.trivi …
    ⊢ ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑(FiberBundle.triviali …
  -/
  set e := trivializationAt F V x₀
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    x₀ : M
    e : Trivialization F Bundle.TotalSpace.proj := FiberBundle.trivializationAt F  …
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ht : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ⊢ ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd := …
  -/
  refine (hs.sub ht).congr_of_eventuallyEq ?_
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    x₀ : M
    e : Trivialization F Bundle.TotalSpace.proj := FiberBundle.trivializationAt F  …
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ht : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ⊢ (nhds x₀).EventuallyEq (fun x => (↑e { proj := x, snd := HSub.hSub (⇑s) (⇑t) …
  -/
  refine eventually_of_mem (e.open_baseSet.mem_nhds <| mem_baseSet_trivializationAt F V x₀) ?_
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    x₀ : M
    e : Trivialization F Bundle.TotalSpace.proj := FiberBundle.trivializationAt F  …
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ht : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ⊢ ∀ (x : M), Membership.mem e.baseSet x → Eq ((fun x => (↑e { proj := x, snd : …
  -/
  intro x hx
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t✝ : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s t : ContMDiffSection I F n V
    x₀ : M
    e : Trivialization F Bundle.TotalSpace.proj := FiberBundle.trivializationAt F  …
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ht : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    x : M
    hx : Membership.mem e.baseSet x
    ⊢ Eq ((fun x => (↑e { proj := x, snd := HSub.hSub (⇑s) (⇑t) x }).2) x) ((fun x …
  -/
  apply (e.linear 𝕜 hx).map_sub
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_sub (s t : Cₛ^n⟮I; F, V⟯) : ⇑(s - t) = s - t :=
  rfl


instance instZero : Zero Cₛ^n⟮I; F, V⟯ :=
  ⟨⟨fun _ => 0, (contMDiff_zeroSection 𝕜 V).of_le le_top⟩⟩


instance inhabited : Inhabited Cₛ^n⟮I; F, V⟯ :=
  ⟨0⟩


@[simp]
theorem coe_zero : ⇑(0 : Cₛ^n⟮I; F, V⟯) = 0 :=
  rfl


instance instNeg : Neg Cₛ^n⟮I; F, V⟯ := by
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    ⊢ Neg (ContMDiffSection I F n V)
  -/
  refine ⟨fun s => ⟨-s, ?_⟩⟩
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s : ContMDiffSection I F n V
    ⊢ ContMDiff I (I.prod (modelWithCornersSelf 𝕜 F)) n fun x => Bundle.TotalSpace …
  -/
  intro x₀
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s : ContMDiffSection I F n V
    x₀ : M
    ⊢ ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.TotalSp …
  -/
  have hs := s.contMDiff x₀
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s : ContMDiffSection I F n V
    x₀ : M
    hs : ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.Tota …
    ⊢ ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.TotalSp …
  -/
  rw [contMDiffAt_section] at hs ⊢
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s : ContMDiffSection I F n V
    x₀ : M
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑(FiberBundle.trivi …
    ⊢ ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑(FiberBundle.triviali …
  -/
  set e := trivializationAt F V x₀
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s : ContMDiffSection I F n V
    x₀ : M
    e : Trivialization F Bundle.TotalSpace.proj := FiberBundle.trivializationAt F  …
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ⊢ ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd := …
  -/
  refine hs.neg.congr_of_eventuallyEq ?_
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s : ContMDiffSection I F n V
    x₀ : M
    e : Trivialization F Bundle.TotalSpace.proj := FiberBundle.trivializationAt F  …
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ⊢ (nhds x₀).EventuallyEq (fun x => (↑e { proj := x, snd := Neg.neg (⇑s) x }).2 …
  -/
  refine eventually_of_mem (e.open_baseSet.mem_nhds <| mem_baseSet_trivializationAt F V x₀) ?_
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s : ContMDiffSection I F n V
    x₀ : M
    e : Trivialization F Bundle.TotalSpace.proj := FiberBundle.trivializationAt F  …
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ⊢ ∀ (x : M), Membership.mem e.baseSet x → Eq ((fun x => (↑e { proj := x, snd : …
  -/
  intro x hx
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s : ContMDiffSection I F n V
    x₀ : M
    e : Trivialization F Bundle.TotalSpace.proj := FiberBundle.trivializationAt F  …
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    x : M
    hx : Membership.mem e.baseSet x
    ⊢ Eq ((fun x => (↑e { proj := x, snd := Neg.neg (⇑s) x }).2) x) ((fun x => Neg …
  -/
  apply (e.linear 𝕜 hx).map_neg
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_neg (s : Cₛ^n⟮I; F, V⟯) : ⇑(-s : Cₛ^n⟮I; F, V⟯) = -s :=
  rfl


instance instNSMul : SMul ℕ Cₛ^n⟮I; F, V⟯ :=
  ⟨nsmulRec⟩


@[simp]
theorem coe_nsmul (s : Cₛ^n⟮I; F, V⟯) (k : ℕ) : ⇑(k • s : Cₛ^n⟮I; F, V⟯) = k • ⇑s := by
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s : ContMDiffSection I F n V
    k : Nat
    ⊢ Eq (⇑(HSMul.hSMul k s)) (HSMul.hSMul k ⇑s)
  -/
  induction' k with k ih
    /-
      case zero
      𝕜 : Type u_1
      inst✝¹³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹² : NormedAddCommGroup E
      inst✝¹¹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹⁰ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : ChartedSpace H M
      F : Type u_5
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : NormedSpace 𝕜 F
      n : ENat
      V : M → Type u_6
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
      inst✝⁴ : (x : M) → TopologicalSpace (V x)
      inst✝³ : FiberBundle F V
      inst✝² : (x : M) → AddCommGroup (V x)
      inst✝¹ : (x : M) → Module 𝕜 (V x)
      inst✝ : VectorBundle 𝕜 F V
      s : ContMDiffSection I F n V
      ⊢ Eq (⇑(HSMul.hSMul 0 s)) (HSMul.hSMul 0 ⇑s)
    -/
  · simp_rw [zero_smul]; rfl
                         /-
                           🎉 no goals
                         -/
  /-
    case succ
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s : ContMDiffSection I F n V
    k : Nat
    ih : Eq (⇑(HSMul.hSMul k s)) (HSMul.hSMul k ⇑s)
    ⊢ Eq (⇑(HSMul.hSMul (HAdd.hAdd k 1) s)) (HSMul.hSMul (HAdd.hAdd k 1) ⇑s)
  -/
  simp_rw [succ_nsmul, ← ih]; rfl
                              /-
                                🎉 no goals
                              -/


instance instZSMul : SMul ℤ Cₛ^n⟮I; F, V⟯ :=
  ⟨zsmulRec⟩


@[simp]
theorem coe_zsmul (s : Cₛ^n⟮I; F, V⟯) (z : ℤ) : ⇑(z • s : Cₛ^n⟮I; F, V⟯) = z • ⇑s := by
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    s : ContMDiffSection I F n V
    z : Int
    ⊢ Eq (⇑(HSMul.hSMul z s)) (HSMul.hSMul z ⇑s)
  -/
  cases' z with n n
    /-
      case ofNat
      𝕜 : Type u_1
      inst✝¹³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹² : NormedAddCommGroup E
      inst✝¹¹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹⁰ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : ChartedSpace H M
      F : Type u_5
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : NormedSpace 𝕜 F
      n✝ : ENat
      V : M → Type u_6
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
      inst✝⁴ : (x : M) → TopologicalSpace (V x)
      inst✝³ : FiberBundle F V
      inst✝² : (x : M) → AddCommGroup (V x)
      inst✝¹ : (x : M) → Module 𝕜 (V x)
      inst✝ : VectorBundle 𝕜 F V
      s : ContMDiffSection I F n✝ V
      n : Nat
      ⊢ Eq (⇑(HSMul.hSMul (Int.ofNat n) s)) (HSMul.hSMul (Int.ofNat n) ⇑s)
    -/
  · refine (coe_nsmul s n).trans ?_
    /-
      case ofNat
      𝕜 : Type u_1
      inst✝¹³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹² : NormedAddCommGroup E
      inst✝¹¹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹⁰ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : ChartedSpace H M
      F : Type u_5
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : NormedSpace 𝕜 F
      n✝ : ENat
      V : M → Type u_6
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
      inst✝⁴ : (x : M) → TopologicalSpace (V x)
      inst✝³ : FiberBundle F V
      inst✝² : (x : M) → AddCommGroup (V x)
      inst✝¹ : (x : M) → Module 𝕜 (V x)
      inst✝ : VectorBundle 𝕜 F V
      s : ContMDiffSection I F n✝ V
      n : Nat
      ⊢ Eq (HSMul.hSMul n ⇑s) (HSMul.hSMul (Int.ofNat n) ⇑s)
    -/
    simp only [Int.ofNat_eq_coe, natCast_zsmul]
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      𝕜 : Type u_1
      inst✝¹³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹² : NormedAddCommGroup E
      inst✝¹¹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹⁰ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : ChartedSpace H M
      F : Type u_5
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : NormedSpace 𝕜 F
      n✝ : ENat
      V : M → Type u_6
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
      inst✝⁴ : (x : M) → TopologicalSpace (V x)
      inst✝³ : FiberBundle F V
      inst✝² : (x : M) → AddCommGroup (V x)
      inst✝¹ : (x : M) → Module 𝕜 (V x)
      inst✝ : VectorBundle 𝕜 F V
      s : ContMDiffSection I F n✝ V
      n : Nat
      ⊢ Eq (⇑(HSMul.hSMul (Int.negSucc n) s)) (HSMul.hSMul (Int.negSucc n) ⇑s)
    -/
  · refine (congr_arg Neg.neg (coe_nsmul s (n + 1))).trans ?_
    /-
      case negSucc
      𝕜 : Type u_1
      inst✝¹³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹² : NormedAddCommGroup E
      inst✝¹¹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹⁰ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : ChartedSpace H M
      F : Type u_5
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : NormedSpace 𝕜 F
      n✝ : ENat
      V : M → Type u_6
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
      inst✝⁴ : (x : M) → TopologicalSpace (V x)
      inst✝³ : FiberBundle F V
      inst✝² : (x : M) → AddCommGroup (V x)
      inst✝¹ : (x : M) → Module 𝕜 (V x)
      inst✝ : VectorBundle 𝕜 F V
      s : ContMDiffSection I F n✝ V
      n : Nat
      ⊢ Eq (Neg.neg (HSMul.hSMul (HAdd.hAdd n 1) ⇑s)) (HSMul.hSMul (Int.negSucc n) ⇑s)
    -/
    simp only [negSucc_zsmul, neg_inj]
    /-
      🎉 no goals
    -/


instance instAddCommGroup : AddCommGroup Cₛ^n⟮I; F, V⟯ :=
  coe_injective.addCommGroup _ coe_zero coe_add coe_neg coe_sub coe_nsmul coe_zsmul


instance instSMul : SMul 𝕜 Cₛ^n⟮I; F, V⟯ := by
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    ⊢ SMul 𝕜 (ContMDiffSection I F n V)
  -/
  refine ⟨fun c s => ⟨c • ⇑s, ?_⟩⟩
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    c : 𝕜
    s : ContMDiffSection I F n V
    ⊢ ContMDiff I (I.prod (modelWithCornersSelf 𝕜 F)) n fun x => Bundle.TotalSpace …
  -/
  intro x₀
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    c : 𝕜
    s : ContMDiffSection I F n V
    x₀ : M
    ⊢ ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.TotalSp …
  -/
  have hs := s.contMDiff x₀
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    c : 𝕜
    s : ContMDiffSection I F n V
    x₀ : M
    hs : ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.Tota …
    ⊢ ContMDiffAt I (I.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle.TotalSp …
  -/
  rw [contMDiffAt_section] at hs ⊢
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    c : 𝕜
    s : ContMDiffSection I F n V
    x₀ : M
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑(FiberBundle.trivi …
    ⊢ ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑(FiberBundle.triviali …
  -/
  set e := trivializationAt F V x₀
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    c : 𝕜
    s : ContMDiffSection I F n V
    x₀ : M
    e : Trivialization F Bundle.TotalSpace.proj := FiberBundle.trivializationAt F  …
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ⊢ ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd := …
  -/
  refine ((contMDiffAt_const (c := c)).smul hs).congr_of_eventuallyEq ?_
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    c : 𝕜
    s : ContMDiffSection I F n V
    x₀ : M
    e : Trivialization F Bundle.TotalSpace.proj := FiberBundle.trivializationAt F  …
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ⊢ (nhds x₀).EventuallyEq (fun x => (↑e { proj := x, snd := HSMul.hSMul c (⇑s)  …
  -/
  refine eventually_of_mem (e.open_baseSet.mem_nhds <| mem_baseSet_trivializationAt F V x₀) ?_
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    c : 𝕜
    s : ContMDiffSection I F n V
    x₀ : M
    e : Trivialization F Bundle.TotalSpace.proj := FiberBundle.trivializationAt F  …
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    ⊢ ∀ (x : M), Membership.mem e.baseSet x → Eq ((fun x => (↑e { proj := x, snd : …
  -/
  intro x hx
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁰ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : ChartedSpace H M
    F : Type u_5
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    n : ENat
    V : M → Type u_6
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F V)
    inst✝⁴ : (x : M) → TopologicalSpace (V x)
    inst✝³ : FiberBundle F V
    s✝ t : ContMDiffSection I F n V
    inst✝² : (x : M) → AddCommGroup (V x)
    inst✝¹ : (x : M) → Module 𝕜 (V x)
    inst✝ : VectorBundle 𝕜 F V
    c : 𝕜
    s : ContMDiffSection I F n V
    x₀ : M
    e : Trivialization F Bundle.TotalSpace.proj := FiberBundle.trivializationAt F  …
    hs : ContMDiffAt I (modelWithCornersSelf 𝕜 F) n (fun x => (↑e { proj := x, snd …
    x : M
    hx : Membership.mem e.baseSet x
    ⊢ Eq ((fun x => (↑e { proj := x, snd := HSMul.hSMul c (⇑s) x }).2) x) ((fun p  …
  -/
  apply (e.linear 𝕜 hx).2
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_smul (r : 𝕜) (s : Cₛ^n⟮I; F, V⟯) : ⇑(r • s : Cₛ^n⟮I; F, V⟯) = r • ⇑s :=
  rfl


variable (I F V n) in
/-- The additive morphism from smooth sections to dependent maps. -/
def coeAddHom : Cₛ^n⟮I; F, V⟯ →+ ∀ x, V x where
  toFun := (↑)
  map_zero' := coe_zero
  map_add' := coe_add


instance instModule : Module 𝕜 Cₛ^n⟮I; F, V⟯ :=
  coe_injective.module 𝕜 (coeAddHom I F n V) coe_smul


protected theorem mdifferentiable' (s : Cₛ^n⟮I; F, V⟯) (hn : 1 ≤ n) :
    MDifferentiable I (I.prod 𝓘(𝕜, F)) fun x => TotalSpace.mk' F x (s x : V x) :=
  s.contMDiff.mdifferentiable hn


protected theorem mdifferentiable (s : Cₛ^∞⟮I; F, V⟯) :
    MDifferentiable I (I.prod 𝓘(𝕜, F)) fun x => TotalSpace.mk' F x (s x : V x) :=
  s.contMDiff.mdifferentiable le_top


protected theorem mdifferentiableAt (s : Cₛ^∞⟮I; F, V⟯) {x} :
    MDifferentiableAt I (I.prod 𝓘(𝕜, F)) (fun x => TotalSpace.mk' F x (s x : V x)) x :=
  s.mdifferentiable x


