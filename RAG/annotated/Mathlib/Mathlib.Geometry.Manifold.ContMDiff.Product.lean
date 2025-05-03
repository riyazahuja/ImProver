theorem ContMDiffWithinAt.prod_mk {f : M → M'} {g : M → N'} (hf : ContMDiffWithinAt I I' n f s x)
    (hg : ContMDiffWithinAt I J' n g s x) :
    ContMDiffWithinAt I (I'.prod J') n (fun x => (f x, g x)) s x := by
  /-
    𝕜 : Type u_1
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁴ : NormedAddCommGroup E
    inst✝¹³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹¹ : TopologicalSpace M
    inst✝¹⁰ : ChartedSpace H M
    E' : Type u_5
    inst✝⁹ : NormedAddCommGroup E'
    inst✝⁸ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁷ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁶ : TopologicalSpace M'
    inst✝⁵ : ChartedSpace H' M'
    F' : Type u_11
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_12
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_13
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    s : Set M
    x : M
    n : ENat
    f : M → M'
    g : M → N'
    hf : ContMDiffWithinAt I I' n f s x
    hg : ContMDiffWithinAt I J' n g s x
    ⊢ ContMDiffWithinAt I (I'.prod J') n (fun x => { fst := f x, snd := g x }) s x
  -/
  rw [contMDiffWithinAt_iff] at *
  /-
    𝕜 : Type u_1
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁴ : NormedAddCommGroup E
    inst✝¹³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹¹ : TopologicalSpace M
    inst✝¹⁰ : ChartedSpace H M
    E' : Type u_5
    inst✝⁹ : NormedAddCommGroup E'
    inst✝⁸ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁷ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁶ : TopologicalSpace M'
    inst✝⁵ : ChartedSpace H' M'
    F' : Type u_11
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_12
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_13
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    s : Set M
    x : M
    n : ENat
    f : M → M'
    g : M → N'
    hf : And (ContinuousWithinAt f s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑ …
    hg : And (ContinuousWithinAt g s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑ …
    ⊢ And (ContinuousWithinAt (fun x => { fst := f x, snd := g x }) s x) (ContDiff …
  -/
  exact ⟨hf.1.prod hg.1, hf.2.prod hg.2⟩
  /-
    🎉 no goals
  -/


theorem ContMDiffWithinAt.prod_mk_space {f : M → E'} {g : M → F'}
    (hf : ContMDiffWithinAt I 𝓘(𝕜, E') n f s x) (hg : ContMDiffWithinAt I 𝓘(𝕜, F') n g s x) :
    ContMDiffWithinAt I 𝓘(𝕜, E' × F') n (fun x => (f x, g x)) s x := by
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    E' : Type u_5
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    F' : Type u_11
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace 𝕜 F'
    s : Set M
    x : M
    n : ENat
    f : M → E'
    g : M → F'
    hf : ContMDiffWithinAt I (modelWithCornersSelf 𝕜 E') n f s x
    hg : ContMDiffWithinAt I (modelWithCornersSelf 𝕜 F') n g s x
    ⊢ ContMDiffWithinAt I (modelWithCornersSelf 𝕜 (Prod E' F')) n (fun x => { fst  …
  -/
  rw [contMDiffWithinAt_iff] at *
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    E' : Type u_5
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    F' : Type u_11
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace 𝕜 F'
    s : Set M
    x : M
    n : ENat
    f : M → E'
    g : M → F'
    hf : And (ContinuousWithinAt f s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑ …
    hg : And (ContinuousWithinAt g s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑ …
    ⊢ And (ContinuousWithinAt (fun x => { fst := f x, snd := g x }) s x) (ContDiff …
  -/
  exact ⟨hf.1.prod hg.1, hf.2.prod hg.2⟩
  /-
    🎉 no goals
  -/


nonrec theorem ContMDiffAt.prod_mk {f : M → M'} {g : M → N'} (hf : ContMDiffAt I I' n f x)
    (hg : ContMDiffAt I J' n g x) : ContMDiffAt I (I'.prod J') n (fun x => (f x, g x)) x :=
  hf.prod_mk hg


nonrec theorem ContMDiffAt.prod_mk_space {f : M → E'} {g : M → F'}
    (hf : ContMDiffAt I 𝓘(𝕜, E') n f x) (hg : ContMDiffAt I 𝓘(𝕜, F') n g x) :
    ContMDiffAt I 𝓘(𝕜, E' × F') n (fun x => (f x, g x)) x :=
  hf.prod_mk_space hg


theorem ContMDiffOn.prod_mk {f : M → M'} {g : M → N'} (hf : ContMDiffOn I I' n f s)
    (hg : ContMDiffOn I J' n g s) : ContMDiffOn I (I'.prod J') n (fun x => (f x, g x)) s :=
  fun x hx => (hf x hx).prod_mk (hg x hx)


theorem ContMDiffOn.prod_mk_space {f : M → E'} {g : M → F'} (hf : ContMDiffOn I 𝓘(𝕜, E') n f s)
    (hg : ContMDiffOn I 𝓘(𝕜, F') n g s) : ContMDiffOn I 𝓘(𝕜, E' × F') n (fun x => (f x, g x)) s :=
  fun x hx => (hf x hx).prod_mk_space (hg x hx)


nonrec theorem ContMDiff.prod_mk {f : M → M'} {g : M → N'} (hf : ContMDiff I I' n f)
    (hg : ContMDiff I J' n g) : ContMDiff I (I'.prod J') n fun x => (f x, g x) := fun x =>
  (hf x).prod_mk (hg x)


theorem ContMDiff.prod_mk_space {f : M → E'} {g : M → F'} (hf : ContMDiff I 𝓘(𝕜, E') n f)
    (hg : ContMDiff I 𝓘(𝕜, F') n g) : ContMDiff I 𝓘(𝕜, E' × F') n fun x => (f x, g x) := fun x =>
  (hf x).prod_mk_space (hg x)


@[deprecated (since := "2024-11-20")] alias SmoothWithinAt.prod_mk := ContMDiffWithinAt.prod_mk


@[deprecated (since := "2024-11-20")]
alias SmoothWithinAt.prod_mk_space := ContMDiffWithinAt.prod_mk_space


@[deprecated (since := "2024-11-20")] alias SmoothAt.prod_mk := ContMDiffAt.prod_mk


@[deprecated (since := "2024-11-20")] alias SmoothAt.prod_mk_space := ContMDiffAt.prod_mk_space


@[deprecated (since := "2024-11-20")] alias SmoothOn.prod_mk := ContMDiffOn.prod_mk


@[deprecated (since := "2024-11-20")] alias SmoothOn.prod_mk_space := ContMDiffOn.prod_mk_space


@[deprecated (since := "2024-11-20")] alias Smooth.prod_mk := ContMDiff.prod_mk


@[deprecated (since := "2024-11-20")] alias Smooth.prod_mk_space := ContMDiff.prod_mk_space


theorem contMDiffWithinAt_fst {s : Set (M × N)} {p : M × N} :
    ContMDiffWithinAt (I.prod J) I n Prod.fst s p := by
  /- porting note: `simp` fails to apply lemmas to `ModelProd`. Was
  rw [contMDiffWithinAt_iff']
  refine' ⟨continuousWithinAt_fst, _⟩
  refine' contDiffWithinAt_fst.congr (fun y hy => _) _
  · simp only [mfld_simps] at hy
    simp only [hy, mfld_simps]
  · simp only [mfld_simps]
  -/
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
    F : Type u_8
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    G : Type u_9
    inst✝² : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_10
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    s : Set (Prod M N)
    p : Prod M N
    ⊢ ContMDiffWithinAt (I.prod J) I n Prod.fst s p
  -/
  rw [contMDiffWithinAt_iff']
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
    F : Type u_8
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    G : Type u_9
    inst✝² : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_10
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    s : Set (Prod M N)
    p : Prod M N
    ⊢ And (ContinuousWithinAt Prod.fst s p) (ContDiffWithinAt 𝕜 (↑n) (Function.com …
  -/
  refine ⟨continuousWithinAt_fst, contDiffWithinAt_fst.congr (fun y hy => ?_) ?_⟩
    /-
      case refine_1
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
      F : Type u_8
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 F
      G : Type u_9
      inst✝² : TopologicalSpace G
      J : ModelWithCorners 𝕜 F G
      N : Type u_10
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace G N
      n : ENat
      s : Set (Prod M N)
      p : Prod M N
      y : Prod E F
      hy : Membership.mem (Inter.inter (extChartAt (I.prod J) p).target (Set.preimag …
      ⊢ Eq (Function.comp (↑(extChartAt I p.1)) (Function.comp Prod.fst ↑(extChartAt …
    -/
  · exact (extChartAt I p.1).right_inv ⟨hy.1.1.1, hy.1.2.1⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
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
      F : Type u_8
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 F
      G : Type u_9
      inst✝² : TopologicalSpace G
      J : ModelWithCorners 𝕜 F G
      N : Type u_10
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace G N
      n : ENat
      s : Set (Prod M N)
      p : Prod M N
      ⊢ Eq (Function.comp (↑(extChartAt I p.1)) (Function.comp Prod.fst ↑(extChartAt …
    -/
  · exact (extChartAt I p.1).right_inv <| (extChartAt I p.1).map_source (mem_extChartAt_source _)
    /-
      🎉 no goals
    -/


theorem ContMDiffWithinAt.fst {f : N → M × M'} {s : Set N} {x : N}
    (hf : ContMDiffWithinAt J (I.prod I') n f s x) :
    ContMDiffWithinAt J I n (fun x => (f x).1) s x :=
  contMDiffWithinAt_fst.comp x hf (mapsTo_image f s)


theorem contMDiffAt_fst {p : M × N} : ContMDiffAt (I.prod J) I n Prod.fst p :=
  contMDiffWithinAt_fst


theorem contMDiffOn_fst {s : Set (M × N)} : ContMDiffOn (I.prod J) I n Prod.fst s := fun _ _ =>
  contMDiffWithinAt_fst


theorem contMDiff_fst : ContMDiff (I.prod J) I n (@Prod.fst M N) := fun _ => contMDiffAt_fst


@[deprecated (since := "2024-11-20")] alias smoothWithinAt_fst := contMDiffWithinAt_fst


@[deprecated (since := "2024-11-20")] alias smoothAt_fst := contMDiffAt_fst


@[deprecated (since := "2024-11-20")] alias smoothOn_fst := contMDiffOn_fst


@[deprecated (since := "2024-11-20")] alias smooth_fst := contMDiff_fst


theorem ContMDiffAt.fst {f : N → M × M'} {x : N} (hf : ContMDiffAt J (I.prod I') n f x) :
    ContMDiffAt J I n (fun x => (f x).1) x :=
  contMDiffAt_fst.comp x hf


theorem ContMDiff.fst {f : N → M × M'} (hf : ContMDiff J (I.prod I') n f) :
    ContMDiff J I n fun x => (f x).1 :=
  contMDiff_fst.comp hf


@[deprecated (since := "2024-11-20")] alias SmoothAt.fst := ContMDiffAt.fst


@[deprecated (since := "2024-11-20")] alias Smooth.fst := ContMDiff.fst


theorem contMDiffWithinAt_snd {s : Set (M × N)} {p : M × N} :
    ContMDiffWithinAt (I.prod J) J n Prod.snd s p := by
  /- porting note: `simp` fails to apply lemmas to `ModelProd`. Was
  rw [contMDiffWithinAt_iff']
  refine' ⟨continuousWithinAt_snd, _⟩
  refine' contDiffWithinAt_snd.congr (fun y hy => _) _
  · simp only [mfld_simps] at hy
    simp only [hy, mfld_simps]
  · simp only [mfld_simps]
  -/
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
    F : Type u_8
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    G : Type u_9
    inst✝² : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_10
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    s : Set (Prod M N)
    p : Prod M N
    ⊢ ContMDiffWithinAt (I.prod J) J n Prod.snd s p
  -/
  rw [contMDiffWithinAt_iff']
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
    F : Type u_8
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    G : Type u_9
    inst✝² : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_10
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    s : Set (Prod M N)
    p : Prod M N
    ⊢ And (ContinuousWithinAt Prod.snd s p) (ContDiffWithinAt 𝕜 (↑n) (Function.com …
  -/
  refine ⟨continuousWithinAt_snd, contDiffWithinAt_snd.congr (fun y hy => ?_) ?_⟩
    /-
      case refine_1
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
      F : Type u_8
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 F
      G : Type u_9
      inst✝² : TopologicalSpace G
      J : ModelWithCorners 𝕜 F G
      N : Type u_10
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace G N
      n : ENat
      s : Set (Prod M N)
      p : Prod M N
      y : Prod E F
      hy : Membership.mem (Inter.inter (extChartAt (I.prod J) p).target (Set.preimag …
      ⊢ Eq (Function.comp (↑(extChartAt J p.2)) (Function.comp Prod.snd ↑(extChartAt …
    -/
  · exact (extChartAt J p.2).right_inv ⟨hy.1.1.2, hy.1.2.2⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
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
      F : Type u_8
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 F
      G : Type u_9
      inst✝² : TopologicalSpace G
      J : ModelWithCorners 𝕜 F G
      N : Type u_10
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace G N
      n : ENat
      s : Set (Prod M N)
      p : Prod M N
      ⊢ Eq (Function.comp (↑(extChartAt J p.2)) (Function.comp Prod.snd ↑(extChartAt …
    -/
  · exact (extChartAt J p.2).right_inv <| (extChartAt J p.2).map_source (mem_extChartAt_source _)
    /-
      🎉 no goals
    -/


theorem ContMDiffWithinAt.snd {f : N → M × M'} {s : Set N} {x : N}
    (hf : ContMDiffWithinAt J (I.prod I') n f s x) :
    ContMDiffWithinAt J I' n (fun x => (f x).2) s x :=
  contMDiffWithinAt_snd.comp x hf (mapsTo_image f s)


theorem contMDiffAt_snd {p : M × N} : ContMDiffAt (I.prod J) J n Prod.snd p :=
  contMDiffWithinAt_snd


theorem contMDiffOn_snd {s : Set (M × N)} : ContMDiffOn (I.prod J) J n Prod.snd s := fun _ _ =>
  contMDiffWithinAt_snd


theorem contMDiff_snd : ContMDiff (I.prod J) J n (@Prod.snd M N) := fun _ => contMDiffAt_snd


@[deprecated (since := "2024-11-20")] alias smoothWithinAt_snd := contMDiffWithinAt_snd


@[deprecated (since := "2024-11-20")] alias smoothAt_snd := contMDiffAt_snd


@[deprecated (since := "2024-11-20")] alias smoothOn_snd := contMDiffOn_snd


@[deprecated (since := "2024-11-20")] alias smooth_snd := contMDiff_snd


theorem ContMDiffAt.snd {f : N → M × M'} {x : N} (hf : ContMDiffAt J (I.prod I') n f x) :
    ContMDiffAt J I' n (fun x => (f x).2) x :=
  contMDiffAt_snd.comp x hf


theorem ContMDiff.snd {f : N → M × M'} (hf : ContMDiff J (I.prod I') n f) :
    ContMDiff J I' n fun x => (f x).2 :=
  contMDiff_snd.comp hf


@[deprecated (since := "2024-11-20")] alias SmoothAt.snd := ContMDiffAt.snd


@[deprecated (since := "2024-11-20")] alias Smooth.snd := ContMDiff.snd


theorem contMDiffWithinAt_prod_iff (f : M → M' × N') :
    ContMDiffWithinAt I (I'.prod J') n f s x ↔
      ContMDiffWithinAt I I' n (Prod.fst ∘ f) s x ∧ ContMDiffWithinAt I J' n (Prod.snd ∘ f) s x :=
  ⟨fun h => ⟨h.fst, h.snd⟩, fun h => h.1.prod_mk h.2⟩


theorem contMDiffWithinAt_prod_module_iff (f : M → F₁ × F₂) :
    ContMDiffWithinAt I 𝓘(𝕜, F₁ × F₂) n f s x ↔
      ContMDiffWithinAt I 𝓘(𝕜, F₁) n (Prod.fst ∘ f) s x ∧
      ContMDiffWithinAt I 𝓘(𝕜, F₂) n (Prod.snd ∘ f) s x := by
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    F₁ : Type u_14
    inst✝³ : NormedAddCommGroup F₁
    inst✝² : NormedSpace 𝕜 F₁
    F₂ : Type u_15
    inst✝¹ : NormedAddCommGroup F₂
    inst✝ : NormedSpace 𝕜 F₂
    s : Set M
    x : M
    n : ENat
    f : M → Prod F₁ F₂
    ⊢ Iff (ContMDiffWithinAt I (modelWithCornersSelf 𝕜 (Prod F₁ F₂)) n f s x) (And …
  -/
  rw [modelWithCornersSelf_prod, ← chartedSpaceSelf_prod]
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    F₁ : Type u_14
    inst✝³ : NormedAddCommGroup F₁
    inst✝² : NormedSpace 𝕜 F₁
    F₂ : Type u_15
    inst✝¹ : NormedAddCommGroup F₂
    inst✝ : NormedSpace 𝕜 F₂
    s : Set M
    x : M
    n : ENat
    f : M → Prod F₁ F₂
    ⊢ Iff (ContMDiffWithinAt I ((modelWithCornersSelf 𝕜 F₁).prod (modelWithCorners …
  -/
  exact contMDiffWithinAt_prod_iff f
  /-
    🎉 no goals
  -/


theorem contMDiffAt_prod_iff (f : M → M' × N') :
    ContMDiffAt I (I'.prod J') n f x ↔
      ContMDiffAt I I' n (Prod.fst ∘ f) x ∧ ContMDiffAt I J' n (Prod.snd ∘ f) x := by
  /-
    𝕜 : Type u_1
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁴ : NormedAddCommGroup E
    inst✝¹³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹¹ : TopologicalSpace M
    inst✝¹⁰ : ChartedSpace H M
    E' : Type u_5
    inst✝⁹ : NormedAddCommGroup E'
    inst✝⁸ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁷ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁶ : TopologicalSpace M'
    inst✝⁵ : ChartedSpace H' M'
    F' : Type u_11
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_12
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_13
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    x : M
    n : ENat
    f : M → Prod M' N'
    ⊢ Iff (ContMDiffAt I (I'.prod J') n f x) (And (ContMDiffAt I I' n (Function.co …
  -/
  simp_rw [← contMDiffWithinAt_univ]; exact contMDiffWithinAt_prod_iff f
                                      /-
                                        🎉 no goals
                                      -/


theorem contMDiffAt_prod_module_iff (f : M → F₁ × F₂) :
    ContMDiffAt I 𝓘(𝕜, F₁ × F₂) n f x ↔
      ContMDiffAt I 𝓘(𝕜, F₁) n (Prod.fst ∘ f) x ∧ ContMDiffAt I 𝓘(𝕜, F₂) n (Prod.snd ∘ f) x := by
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    F₁ : Type u_14
    inst✝³ : NormedAddCommGroup F₁
    inst✝² : NormedSpace 𝕜 F₁
    F₂ : Type u_15
    inst✝¹ : NormedAddCommGroup F₂
    inst✝ : NormedSpace 𝕜 F₂
    x : M
    n : ENat
    f : M → Prod F₁ F₂
    ⊢ Iff (ContMDiffAt I (modelWithCornersSelf 𝕜 (Prod F₁ F₂)) n f x) (And (ContMD …
  -/
  rw [modelWithCornersSelf_prod, ← chartedSpaceSelf_prod]
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    F₁ : Type u_14
    inst✝³ : NormedAddCommGroup F₁
    inst✝² : NormedSpace 𝕜 F₁
    F₂ : Type u_15
    inst✝¹ : NormedAddCommGroup F₂
    inst✝ : NormedSpace 𝕜 F₂
    x : M
    n : ENat
    f : M → Prod F₁ F₂
    ⊢ Iff (ContMDiffAt I ((modelWithCornersSelf 𝕜 F₁).prod (modelWithCornersSelf 𝕜 …
  -/
  exact contMDiffAt_prod_iff f
  /-
    🎉 no goals
  -/


theorem contMDiffOn_prod_iff (f : M → M' × N') :
    ContMDiffOn I (I'.prod J') n f s ↔
      ContMDiffOn I I' n (Prod.fst ∘ f) s ∧ ContMDiffOn I J' n (Prod.snd ∘ f) s :=
  ⟨fun h ↦ ⟨fun x hx ↦ ((contMDiffWithinAt_prod_iff f).1 (h x hx)).1,
      fun x hx ↦ ((contMDiffWithinAt_prod_iff f).1 (h x hx)).2⟩ ,
    fun h x hx ↦ (contMDiffWithinAt_prod_iff f).2 ⟨h.1 x hx, h.2 x hx⟩⟩


theorem contMDiffOn_prod_module_iff (f : M → F₁ × F₂) :
    ContMDiffOn I 𝓘(𝕜, F₁ × F₂) n f s ↔
      ContMDiffOn I 𝓘(𝕜, F₁) n (Prod.fst ∘ f) s ∧ ContMDiffOn I 𝓘(𝕜, F₂) n (Prod.snd ∘ f) s := by
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    F₁ : Type u_14
    inst✝³ : NormedAddCommGroup F₁
    inst✝² : NormedSpace 𝕜 F₁
    F₂ : Type u_15
    inst✝¹ : NormedAddCommGroup F₂
    inst✝ : NormedSpace 𝕜 F₂
    s : Set M
    n : ENat
    f : M → Prod F₁ F₂
    ⊢ Iff (ContMDiffOn I (modelWithCornersSelf 𝕜 (Prod F₁ F₂)) n f s) (And (ContMD …
  -/
  rw [modelWithCornersSelf_prod, ← chartedSpaceSelf_prod]
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    F₁ : Type u_14
    inst✝³ : NormedAddCommGroup F₁
    inst✝² : NormedSpace 𝕜 F₁
    F₂ : Type u_15
    inst✝¹ : NormedAddCommGroup F₂
    inst✝ : NormedSpace 𝕜 F₂
    s : Set M
    n : ENat
    f : M → Prod F₁ F₂
    ⊢ Iff (ContMDiffOn I ((modelWithCornersSelf 𝕜 F₁).prod (modelWithCornersSelf 𝕜 …
  -/
  exact contMDiffOn_prod_iff f
  /-
    🎉 no goals
  -/


theorem contMDiff_prod_iff (f : M → M' × N') :
    ContMDiff I (I'.prod J') n f ↔
      ContMDiff I I' n (Prod.fst ∘ f) ∧ ContMDiff I J' n (Prod.snd ∘ f) :=
                                        /-
                                          𝕜 : Type u_1
                                          inst✝¹⁵ : NontriviallyNormedField 𝕜
                                          E : Type u_2
                                          inst✝¹⁴ : NormedAddCommGroup E
                                          inst✝¹³ : NormedSpace 𝕜 E
                                          H : Type u_3
                                          inst✝¹² : TopologicalSpace H
                                          I : ModelWithCorners 𝕜 E H
                                          M : Type u_4
                                          inst✝¹¹ : TopologicalSpace M
                                          inst✝¹⁰ : ChartedSpace H M
                                          E' : Type u_5
                                          inst✝⁹ : NormedAddCommGroup E'
                                          inst✝⁸ : NormedSpace 𝕜 E'
                                          H' : Type u_6
                                          inst✝⁷ : TopologicalSpace H'
                                          I' : ModelWithCorners 𝕜 E' H'
                                          M' : Type u_7
                                          inst✝⁶ : TopologicalSpace M'
                                          inst✝⁵ : ChartedSpace H' M'
                                          F' : Type u_11
                                          inst✝⁴ : NormedAddCommGroup F'
                                          inst✝³ : NormedSpace 𝕜 F'
                                          G' : Type u_12
                                          inst✝² : TopologicalSpace G'
                                          J' : ModelWithCorners 𝕜 F' G'
                                          N' : Type u_13
                                          inst✝¹ : TopologicalSpace N'
                                          inst✝ : ChartedSpace G' N'
                                          n : ENat
                                          f : M → Prod M' N'
                                          h : And (ContMDiff I I' n (Function.comp Prod.fst f)) (ContMDiff I J' n (Funct …
                                          ⊢ ContMDiff I (I'.prod J') n f
                                        -/
  ⟨fun h => ⟨h.fst, h.snd⟩, fun h => by convert h.1.prod_mk h.2⟩
                                        /-
                                          🎉 no goals
                                        -/


theorem contMDiff_prod_module_iff (f : M → F₁ × F₂) :
    ContMDiff I 𝓘(𝕜, F₁ × F₂) n f ↔
      ContMDiff I 𝓘(𝕜, F₁) n (Prod.fst ∘ f) ∧ ContMDiff I 𝓘(𝕜, F₂) n (Prod.snd ∘ f) := by
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    F₁ : Type u_14
    inst✝³ : NormedAddCommGroup F₁
    inst✝² : NormedSpace 𝕜 F₁
    F₂ : Type u_15
    inst✝¹ : NormedAddCommGroup F₂
    inst✝ : NormedSpace 𝕜 F₂
    n : ENat
    f : M → Prod F₁ F₂
    ⊢ Iff (ContMDiff I (modelWithCornersSelf 𝕜 (Prod F₁ F₂)) n f) (And (ContMDiff  …
  -/
  rw [modelWithCornersSelf_prod, ← chartedSpaceSelf_prod]
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    F₁ : Type u_14
    inst✝³ : NormedAddCommGroup F₁
    inst✝² : NormedSpace 𝕜 F₁
    F₂ : Type u_15
    inst✝¹ : NormedAddCommGroup F₂
    inst✝ : NormedSpace 𝕜 F₂
    n : ENat
    f : M → Prod F₁ F₂
    ⊢ Iff (ContMDiff I ((modelWithCornersSelf 𝕜 F₁).prod (modelWithCornersSelf 𝕜 F …
  -/
  exact contMDiff_prod_iff f
  /-
    🎉 no goals
  -/


theorem contMDiff_prod_assoc :
    ContMDiff ((I.prod I').prod J) (I.prod (I'.prod J)) n
      fun x : (M × M') × N => (x.1.1, x.1.2, x.2) :=
  contMDiff_fst.fst.prod_mk <| contMDiff_fst.snd.prod_mk contMDiff_snd


@[deprecated (since := "2024-11-20")] alias smoothAt_prod_iff := contMDiffAt_prod_iff


@[deprecated (since := "2024-11-20")] alias smooth_prod_iff := contMDiff_prod_iff


@[deprecated (since := "2024-11-20")] alias smooth_prod_assoc := contMDiff_prod_assoc


/-- The product map of two `C^n` functions within a set at a point is `C^n`
within the product set at the product point. -/
theorem ContMDiffWithinAt.prod_map' {p : M × N} (hf : ContMDiffWithinAt I I' n f s p.1)
    (hg : ContMDiffWithinAt J J' n g r p.2) :
    ContMDiffWithinAt (I.prod J) (I'.prod J') n (Prod.map f g) (s ×ˢ r) p :=
  (hf.comp p contMDiffWithinAt_fst (prod_subset_preimage_fst _ _)).prod_mk <|
    hg.comp p contMDiffWithinAt_snd (prod_subset_preimage_snd _ _)


theorem ContMDiffWithinAt.prod_map (hf : ContMDiffWithinAt I I' n f s x)
    (hg : ContMDiffWithinAt J J' n g r y) :
    ContMDiffWithinAt (I.prod J) (I'.prod J') n (Prod.map f g) (s ×ˢ r) (x, y) :=
  ContMDiffWithinAt.prod_map' hf hg


theorem ContMDiffAt.prod_map (hf : ContMDiffAt I I' n f x) (hg : ContMDiffAt J J' n g y) :
    ContMDiffAt (I.prod J) (I'.prod J') n (Prod.map f g) (x, y) := by
  /-
    𝕜 : Type u_1
    inst✝²⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹⁶ : TopologicalSpace M
    inst✝¹⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝¹² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹¹ : TopologicalSpace M'
    inst✝¹⁰ : ChartedSpace H' M'
    F : Type u_8
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    G : Type u_9
    inst✝⁷ : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_10
    inst✝⁶ : TopologicalSpace N
    inst✝⁵ : ChartedSpace G N
    F' : Type u_11
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_12
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_13
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    f : M → M'
    x : M
    n : ENat
    g : N → N'
    y : N
    hf : ContMDiffAt I I' n f x
    hg : ContMDiffAt J J' n g y
    ⊢ ContMDiffAt (I.prod J) (I'.prod J') n (Prod.map f g) { fst := x, snd := y }
  -/
  rw [← contMDiffWithinAt_univ] at *
  /-
    𝕜 : Type u_1
    inst✝²⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹⁶ : TopologicalSpace M
    inst✝¹⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝¹² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹¹ : TopologicalSpace M'
    inst✝¹⁰ : ChartedSpace H' M'
    F : Type u_8
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    G : Type u_9
    inst✝⁷ : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_10
    inst✝⁶ : TopologicalSpace N
    inst✝⁵ : ChartedSpace G N
    F' : Type u_11
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_12
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_13
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    f : M → M'
    x : M
    n : ENat
    g : N → N'
    y : N
    hf : ContMDiffWithinAt I I' n f Set.univ x
    hg : ContMDiffWithinAt J J' n g Set.univ y
    ⊢ ContMDiffWithinAt (I.prod J) (I'.prod J') n (Prod.map f g) Set.univ { fst := …
  -/
  convert hf.prod_map hg
  /-
    case h.e'_23
    𝕜 : Type u_1
    inst✝²⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹⁶ : TopologicalSpace M
    inst✝¹⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝¹² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹¹ : TopologicalSpace M'
    inst✝¹⁰ : ChartedSpace H' M'
    F : Type u_8
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    G : Type u_9
    inst✝⁷ : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_10
    inst✝⁶ : TopologicalSpace N
    inst✝⁵ : ChartedSpace G N
    F' : Type u_11
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_12
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_13
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    f : M → M'
    x : M
    n : ENat
    g : N → N'
    y : N
    hf : ContMDiffWithinAt I I' n f Set.univ x
    hg : ContMDiffWithinAt J J' n g Set.univ y
    ⊢ Eq Set.univ (SProd.sprod Set.univ Set.univ)
  -/
  exact univ_prod_univ.symm
  /-
    🎉 no goals
  -/


theorem ContMDiffAt.prod_map' {p : M × N} (hf : ContMDiffAt I I' n f p.1)
    (hg : ContMDiffAt J J' n g p.2) : ContMDiffAt (I.prod J) (I'.prod J') n (Prod.map f g) p := by
  /-
    𝕜 : Type u_1
    inst✝²⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹⁶ : TopologicalSpace M
    inst✝¹⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝¹² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹¹ : TopologicalSpace M'
    inst✝¹⁰ : ChartedSpace H' M'
    F : Type u_8
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    G : Type u_9
    inst✝⁷ : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_10
    inst✝⁶ : TopologicalSpace N
    inst✝⁵ : ChartedSpace G N
    F' : Type u_11
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_12
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_13
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    f : M → M'
    n : ENat
    g : N → N'
    p : Prod M N
    hf : ContMDiffAt I I' n f p.1
    hg : ContMDiffAt J J' n g p.2
    ⊢ ContMDiffAt (I.prod J) (I'.prod J') n (Prod.map f g) p
  -/
  rcases p with ⟨⟩
  /-
    case mk
    𝕜 : Type u_1
    inst✝²⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹⁶ : TopologicalSpace M
    inst✝¹⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝¹² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹¹ : TopologicalSpace M'
    inst✝¹⁰ : ChartedSpace H' M'
    F : Type u_8
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    G : Type u_9
    inst✝⁷ : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_10
    inst✝⁶ : TopologicalSpace N
    inst✝⁵ : ChartedSpace G N
    F' : Type u_11
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_12
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_13
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    f : M → M'
    n : ENat
    g : N → N'
    fst✝ : M
    snd✝ : N
    hf : ContMDiffAt I I' n f { fst := fst✝, snd := snd✝ }.1
    hg : ContMDiffAt J J' n g { fst := fst✝, snd := snd✝ }.2
    ⊢ ContMDiffAt (I.prod J) (I'.prod J') n (Prod.map f g) { fst := fst✝, snd := s …
  -/
  exact hf.prod_map hg
  /-
    🎉 no goals
  -/


theorem ContMDiffOn.prod_map (hf : ContMDiffOn I I' n f s) (hg : ContMDiffOn J J' n g r) :
    ContMDiffOn (I.prod J) (I'.prod J') n (Prod.map f g) (s ×ˢ r) :=
  (hf.comp contMDiffOn_fst (prod_subset_preimage_fst _ _)).prod_mk <|
    hg.comp contMDiffOn_snd (prod_subset_preimage_snd _ _)


theorem ContMDiff.prod_map (hf : ContMDiff I I' n f) (hg : ContMDiff J J' n g) :
    ContMDiff (I.prod J) (I'.prod J') n (Prod.map f g) := by
  /-
    𝕜 : Type u_1
    inst✝²⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹⁶ : TopologicalSpace M
    inst✝¹⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝¹² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹¹ : TopologicalSpace M'
    inst✝¹⁰ : ChartedSpace H' M'
    F : Type u_8
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    G : Type u_9
    inst✝⁷ : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_10
    inst✝⁶ : TopologicalSpace N
    inst✝⁵ : ChartedSpace G N
    F' : Type u_11
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_12
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_13
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    f : M → M'
    n : ENat
    g : N → N'
    hf : ContMDiff I I' n f
    hg : ContMDiff J J' n g
    ⊢ ContMDiff (I.prod J) (I'.prod J') n (Prod.map f g)
  -/
  intro p
  /-
    𝕜 : Type u_1
    inst✝²⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹⁶ : TopologicalSpace M
    inst✝¹⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝¹² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹¹ : TopologicalSpace M'
    inst✝¹⁰ : ChartedSpace H' M'
    F : Type u_8
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    G : Type u_9
    inst✝⁷ : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_10
    inst✝⁶ : TopologicalSpace N
    inst✝⁵ : ChartedSpace G N
    F' : Type u_11
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_12
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_13
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    f : M → M'
    n : ENat
    g : N → N'
    hf : ContMDiff I I' n f
    hg : ContMDiff J J' n g
    p : Prod M N
    ⊢ ContMDiffAt (I.prod J) (I'.prod J') n (Prod.map f g) p
  -/
  exact (hf p.1).prod_map' (hg p.2)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-20")] alias SmoothWithinAt.prod_map := ContMDiffWithinAt.prod_map


@[deprecated (since := "2024-11-20")] alias SmoothAt.prod_map := ContMDiffAt.prod_map


@[deprecated (since := "2024-11-20")] alias SmoothOn.prod_map := ContMDiffOn.prod_map


@[deprecated (since := "2024-11-20")] alias Smooth.prod_map := ContMDiff.prod_map


theorem contMDiffWithinAt_pi_space :
    ContMDiffWithinAt I 𝓘(𝕜, ∀ i, Fi i) n φ s x ↔
      ∀ i, ContMDiffWithinAt I 𝓘(𝕜, Fi i) n (fun x => φ x i) s x := by
  simp only [contMDiffWithinAt_iff, continuousWithinAt_pi, contDiffWithinAt_pi, forall_and,
    writtenInExtChartAt, extChartAt_model_space_eq_id, Function.comp_def, PartialEquiv.refl_coe, id]


theorem contMDiffOn_pi_space :
    ContMDiffOn I 𝓘(𝕜, ∀ i, Fi i) n φ s ↔ ∀ i, ContMDiffOn I 𝓘(𝕜, Fi i) n (fun x => φ x i) s :=
  ⟨fun h i x hx => contMDiffWithinAt_pi_space.1 (h x hx) i, fun h x hx =>
    contMDiffWithinAt_pi_space.2 fun i => h i x hx⟩


theorem contMDiffAt_pi_space :
    ContMDiffAt I 𝓘(𝕜, ∀ i, Fi i) n φ x ↔ ∀ i, ContMDiffAt I 𝓘(𝕜, Fi i) n (fun x => φ x i) x :=
  contMDiffWithinAt_pi_space


theorem contMDiff_pi_space :
    ContMDiff I 𝓘(𝕜, ∀ i, Fi i) n φ ↔ ∀ i, ContMDiff I 𝓘(𝕜, Fi i) n fun x => φ x i :=
  ⟨fun h i x => contMDiffAt_pi_space.1 (h x) i, fun h x => contMDiffAt_pi_space.2 fun i => h i x⟩


@[deprecated (since := "2024-11-20")] alias smoothWithinAt_pi_space := contMDiffWithinAt_pi_space


@[deprecated (since := "2024-11-20")] alias smoothAt_pi_space := contMDiffAt_pi_space


@[deprecated (since := "2024-11-20")] alias smoothOn_pi_space := contMDiffOn_pi_space


@[deprecated (since := "2024-11-20")] alias smooth_pi_space := contMDiff_pi_space


