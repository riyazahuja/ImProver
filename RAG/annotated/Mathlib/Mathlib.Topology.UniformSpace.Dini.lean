/-- **Dini's theorem** If `F n` is a monotone increasing collection of continuous functions
converging pointwise to a continuous function `f`, then `F n` converges locally uniformly to `f`. -/
lemma tendstoLocallyUniformly_of_forall_tendsto
    (hF_cont : ∀ i, Continuous (F i)) (hF_mono : Monotone F) (hf : Continuous f)
    (h_tendsto : ∀ x, Tendsto (F · x) atTop (𝓝 (f x))) :
    TendstoLocallyUniformly F f atTop := by
  /-
    ι : Type u_1
    α : Type u_2
    G : Type u_3
    inst✝² : Preorder ι
    inst✝¹ : TopologicalSpace α
    inst✝ : NormedLatticeAddCommGroup G
    F : ι → α → G
    f : α → G
    hF_cont : ∀ (i : ι), Continuous (F i)
    hF_mono : Monotone F
    hf : Continuous f
    h_tendsto : ∀ (x : α), Filter.Tendsto (fun x_1 => F x_1 x) Filter.atTop (nhds  …
    ⊢ TendstoLocallyUniformly F f Filter.atTop
  -/
  refine (atTop : Filter ι).eq_or_neBot.elim (fun h ↦ ?eq_bot) (fun _ ↦ ?_)
  /-
    case eq_bot
    ι : Type u_1
    α : Type u_2
    G : Type u_3
    inst✝² : Preorder ι
    inst✝¹ : TopologicalSpace α
    inst✝ : NormedLatticeAddCommGroup G
    F : ι → α → G
    f : α → G
    hF_cont : ∀ (i : ι), Continuous (F i)
    hF_mono : Monotone F
    hf : Continuous f
    h_tendsto : ∀ (x : α), Filter.Tendsto (fun x_1 => F x_1 x) Filter.atTop (nhds  …
    h : Eq Filter.atTop Bot.bot
    ⊢ TendstoLocallyUniformly F f Filter.atTop
  -/
  case eq_bot => simp [h, tendstoLocallyUniformly_iff_forall_tendsto]
  have F_le_f (x : α) (n : ι) : F n x ≤ f x := by
    refine ge_of_tendsto (h_tendsto x) ?_
    filter_upwards [Ici_mem_atTop n] with m hnm
    exact hF_mono hnm x
  /-
    ι : Type u_1
    α : Type u_2
    G : Type u_3
    inst✝² : Preorder ι
    inst✝¹ : TopologicalSpace α
    inst✝ : NormedLatticeAddCommGroup G
    F : ι → α → G
    f : α → G
    hF_cont : ∀ (i : ι), Continuous (F i)
    hF_mono : Monotone F
    hf : Continuous f
    h_tendsto : ∀ (x : α), Filter.Tendsto (fun x_1 => F x_1 x) Filter.atTop (nhds  …
    x✝ : Filter.atTop.NeBot
    F_le_f : ∀ (x : α) (n : ι), LE.le (F n x) (f x)
    ⊢ TendstoLocallyUniformly F f Filter.atTop
  -/
  simp_rw [Metric.tendstoLocallyUniformly_iff, dist_eq_norm']
  /-
    ι : Type u_1
    α : Type u_2
    G : Type u_3
    inst✝² : Preorder ι
    inst✝¹ : TopologicalSpace α
    inst✝ : NormedLatticeAddCommGroup G
    F : ι → α → G
    f : α → G
    hF_cont : ∀ (i : ι), Continuous (F i)
    hF_mono : Monotone F
    hf : Continuous f
    h_tendsto : ∀ (x : α), Filter.Tendsto (fun x_1 => F x_1 x) Filter.atTop (nhds  …
    x✝ : Filter.atTop.NeBot
    F_le_f : ∀ (x : α) (n : ι), LE.le (F n x) (f x)
    ⊢ ∀ (ε : Real), GT.gt ε 0 → ∀ (x : α), Exists fun t => And (Membership.mem (nh …
  -/
  intro ε ε_pos x
  /-
    ι : Type u_1
    α : Type u_2
    G : Type u_3
    inst✝² : Preorder ι
    inst✝¹ : TopologicalSpace α
    inst✝ : NormedLatticeAddCommGroup G
    F : ι → α → G
    f : α → G
    hF_cont : ∀ (i : ι), Continuous (F i)
    hF_mono : Monotone F
    hf : Continuous f
    h_tendsto : ∀ (x : α), Filter.Tendsto (fun x_1 => F x_1 x) Filter.atTop (nhds  …
    x✝ : Filter.atTop.NeBot
    F_le_f : ∀ (x : α) (n : ι), LE.le (F n x) (f x)
    ε : Real
    ε_pos : GT.gt ε 0
    x : α
    ⊢ Exists fun t => And (Membership.mem (nhds x) t) (Filter.Eventually (fun n => …
  -/
  simp_rw +singlePass [tendsto_iff_norm_sub_tendsto_zero] at h_tendsto
  /-
    ι : Type u_1
    α : Type u_2
    G : Type u_3
    inst✝² : Preorder ι
    inst✝¹ : TopologicalSpace α
    inst✝ : NormedLatticeAddCommGroup G
    F : ι → α → G
    f : α → G
    hF_cont : ∀ (i : ι), Continuous (F i)
    hF_mono : Monotone F
    hf : Continuous f
    x✝ : Filter.atTop.NeBot
    F_le_f : ∀ (x : α) (n : ι), LE.le (F n x) (f x)
    ε : Real
    ε_pos : GT.gt ε 0
    x : α
    h_tendsto : ∀ (x : α), Filter.Tendsto (fun e => Norm.norm (HSub.hSub (F e x) ( …
    ⊢ Exists fun t => And (Membership.mem (nhds x) t) (Filter.Eventually (fun n => …
  -/
  obtain ⟨n, hn⟩ := (h_tendsto x).eventually (eventually_lt_nhds ε_pos) |>.exists
  /-
    case intro
    ι : Type u_1
    α : Type u_2
    G : Type u_3
    inst✝² : Preorder ι
    inst✝¹ : TopologicalSpace α
    inst✝ : NormedLatticeAddCommGroup G
    F : ι → α → G
    f : α → G
    hF_cont : ∀ (i : ι), Continuous (F i)
    hF_mono : Monotone F
    hf : Continuous f
    x✝ : Filter.atTop.NeBot
    F_le_f : ∀ (x : α) (n : ι), LE.le (F n x) (f x)
    ε : Real
    ε_pos : GT.gt ε 0
    x : α
    h_tendsto : ∀ (x : α), Filter.Tendsto (fun e => Norm.norm (HSub.hSub (F e x) ( …
    n : ι
    hn : LT.lt (Norm.norm (HSub.hSub (F n x) (f x))) ε
    ⊢ Exists fun t => And (Membership.mem (nhds x) t) (Filter.Eventually (fun n => …
  -/
  refine ⟨{y | ‖F n y - f y‖ < ε}, ⟨isOpen_lt (by fun_prop) continuous_const |>.mem_nhds hn, ?_⟩⟩
  /-
    case intro
    ι : Type u_1
    α : Type u_2
    G : Type u_3
    inst✝² : Preorder ι
    inst✝¹ : TopologicalSpace α
    inst✝ : NormedLatticeAddCommGroup G
    F : ι → α → G
    f : α → G
    hF_cont : ∀ (i : ι), Continuous (F i)
    hF_mono : Monotone F
    hf : Continuous f
    x✝ : Filter.atTop.NeBot
    F_le_f : ∀ (x : α) (n : ι), LE.le (F n x) (f x)
    ε : Real
    ε_pos : GT.gt ε 0
    x : α
    h_tendsto : ∀ (x : α), Filter.Tendsto (fun e => Norm.norm (HSub.hSub (F e x) ( …
    n : ι
    hn : LT.lt (Norm.norm (HSub.hSub (F n x) (f x))) ε
    ⊢ Filter.Eventually (fun n_1 => ∀ (y : α), Membership.mem (setOf fun y => LT.l …
  -/
  filter_upwards [eventually_ge_atTop n] with m hnm z hz
  /-
    case h
    ι : Type u_1
    α : Type u_2
    G : Type u_3
    inst✝² : Preorder ι
    inst✝¹ : TopologicalSpace α
    inst✝ : NormedLatticeAddCommGroup G
    F : ι → α → G
    f : α → G
    hF_cont : ∀ (i : ι), Continuous (F i)
    hF_mono : Monotone F
    hf : Continuous f
    x✝ : Filter.atTop.NeBot
    F_le_f : ∀ (x : α) (n : ι), LE.le (F n x) (f x)
    ε : Real
    ε_pos : GT.gt ε 0
    x : α
    h_tendsto : ∀ (x : α), Filter.Tendsto (fun e => Norm.norm (HSub.hSub (F e x) ( …
    n : ι
    hn : LT.lt (Norm.norm (HSub.hSub (F n x) (f x))) ε
    m : ι
    hnm : LE.le n m
    z : α
    hz : LT.lt (Norm.norm (HSub.hSub (F n z) (f z))) ε
    ⊢ LT.lt (Norm.norm (HSub.hSub (F m z) (f z))) ε
  -/
  refine norm_le_norm_of_abs_le_abs ?_ |>.trans_lt hz
  /-
    case h
    ι : Type u_1
    α : Type u_2
    G : Type u_3
    inst✝² : Preorder ι
    inst✝¹ : TopologicalSpace α
    inst✝ : NormedLatticeAddCommGroup G
    F : ι → α → G
    f : α → G
    hF_cont : ∀ (i : ι), Continuous (F i)
    hF_mono : Monotone F
    hf : Continuous f
    x✝ : Filter.atTop.NeBot
    F_le_f : ∀ (x : α) (n : ι), LE.le (F n x) (f x)
    ε : Real
    ε_pos : GT.gt ε 0
    x : α
    h_tendsto : ∀ (x : α), Filter.Tendsto (fun e => Norm.norm (HSub.hSub (F e x) ( …
    n : ι
    hn : LT.lt (Norm.norm (HSub.hSub (F n x) (f x))) ε
    m : ι
    hnm : LE.le n m
    z : α
    hz : LT.lt (Norm.norm (HSub.hSub (F n z) (f z))) ε
    ⊢ LE.le (abs (HSub.hSub (F m z) (f z))) (abs (HSub.hSub (F n z) (f z)))
  -/
  simp only [abs_of_nonpos (sub_nonpos_of_le (F_le_f _ _)), neg_sub, sub_le_sub_iff_left]
  /-
    case h
    ι : Type u_1
    α : Type u_2
    G : Type u_3
    inst✝² : Preorder ι
    inst✝¹ : TopologicalSpace α
    inst✝ : NormedLatticeAddCommGroup G
    F : ι → α → G
    f : α → G
    hF_cont : ∀ (i : ι), Continuous (F i)
    hF_mono : Monotone F
    hf : Continuous f
    x✝ : Filter.atTop.NeBot
    F_le_f : ∀ (x : α) (n : ι), LE.le (F n x) (f x)
    ε : Real
    ε_pos : GT.gt ε 0
    x : α
    h_tendsto : ∀ (x : α), Filter.Tendsto (fun e => Norm.norm (HSub.hSub (F e x) ( …
    n : ι
    hn : LT.lt (Norm.norm (HSub.hSub (F n x) (f x))) ε
    m : ι
    hnm : LE.le n m
    z : α
    hz : LT.lt (Norm.norm (HSub.hSub (F n z) (f z))) ε
    ⊢ LE.le (F n z) (F m z)
  -/
  exact hF_mono hnm z
  /-
    🎉 no goals
  -/


/-- **Dini's theorem** If `F n` is a monotone increasing collection of continuous functions on a
set `s` converging pointwise to a continuous function `f`, then `F n` converges locally uniformly
to `f`. -/
lemma tendstoLocallyUniformlyOn_of_forall_tendsto {s : Set α}
    (hF_cont : ∀ i, ContinuousOn (F i) s) (hF_mono : ∀ x ∈ s, Monotone (F · x))
    (hf : ContinuousOn f s) (h_tendsto : ∀ x ∈ s, Tendsto (F · x) atTop (𝓝 (f x))) :
    TendstoLocallyUniformlyOn F f atTop s := by
  /-
    ι : Type u_1
    α : Type u_2
    G : Type u_3
    inst✝² : Preorder ι
    inst✝¹ : TopologicalSpace α
    inst✝ : NormedLatticeAddCommGroup G
    F : ι → α → G
    f : α → G
    s : Set α
    hF_cont : ∀ (i : ι), ContinuousOn (F i) s
    hF_mono : ∀ (x : α), Membership.mem s x → Monotone fun x_1 => F x_1 x
    hf : ContinuousOn f s
    h_tendsto : ∀ (x : α), Membership.mem s x → Filter.Tendsto (fun x_1 => F x_1 x …
    ⊢ TendstoLocallyUniformlyOn F f Filter.atTop s
  -/
  rw [tendstoLocallyUniformlyOn_iff_tendstoLocallyUniformly_comp_coe]
  exact tendstoLocallyUniformly_of_forall_tendsto (hF_cont · |>.restrict)
    (fun _ _ h x ↦ hF_mono _ x.2 h) hf.restrict (fun x ↦ h_tendsto x x.2)


/-- **Dini's theorem** If `F n` is a monotone increasing collection of continuous functions on a
compact space converging pointwise to a continuous function `f`, then `F n` converges uniformly to
`f`. -/
lemma tendstoUniformly_of_forall_tendsto [CompactSpace α] (hF_cont : ∀ i, Continuous (F i))
    (hF_mono : Monotone F) (hf : Continuous f) (h_tendsto : ∀ x, Tendsto (F · x) atTop (𝓝 (f x))) :
    TendstoUniformly F f atTop :=
  tendstoLocallyUniformly_iff_tendstoUniformly_of_compactSpace.mp <|
    tendstoLocallyUniformly_of_forall_tendsto hF_cont hF_mono hf h_tendsto


/-- **Dini's theorem** If `F n` is a monotone increasing collection of continuous functions on a
compact set `s` converging pointwise to a continuous function `f`, then `F n` converges uniformly
to `f`. -/
lemma tendstoUniformlyOn_of_forall_tendsto {s : Set α} (hs : IsCompact s)
    (hF_cont : ∀ i, ContinuousOn (F i) s) (hF_mono : ∀ x ∈ s, Monotone (F · x))
    (hf : ContinuousOn f s) (h_tendsto : ∀ x ∈ s, Tendsto (F · x) atTop (𝓝 (f x))) :
    TendstoUniformlyOn F f atTop s :=
  tendstoLocallyUniformlyOn_iff_tendstoUniformlyOn_of_compact hs |>.mp <|
    tendstoLocallyUniformlyOn_of_forall_tendsto hF_cont hF_mono hf h_tendsto


/-- **Dini's theorem** If `F n` is a monotone decreasing collection of continuous functions on a
converging pointwise to a continuous function `f`, then `F n` converges locally uniformly to `f`. -/
lemma tendstoLocallyUniformly_of_forall_tendsto
    (hF_cont : ∀ i, Continuous (F i)) (hF_anti : Antitone F) (hf : Continuous f)
    (h_tendsto : ∀ x, Tendsto (F · x) atTop (𝓝 (f x))) :
    TendstoLocallyUniformly F f atTop :=
  Monotone.tendstoLocallyUniformly_of_forall_tendsto (G := Gᵒᵈ) hF_cont hF_anti hf h_tendsto


/-- **Dini's theorem** If `F n` is a monotone decreasing collection of continuous functions on a
set `s` converging pointwise to a continuous function `f`, then `F n` converges locally uniformly
to `f`. -/
lemma tendstoLocallyUniformlyOn_of_forall_tendsto {s : Set α}
    (hF_cont : ∀ i, ContinuousOn (F i) s) (hF_anti : ∀ x ∈ s, Antitone (F · x))
    (hf : ContinuousOn f s) (h_tendsto : ∀ x ∈ s, Tendsto (F · x) atTop (𝓝 (f x))) :
    TendstoLocallyUniformlyOn F f atTop s :=
  Monotone.tendstoLocallyUniformlyOn_of_forall_tendsto (G := Gᵒᵈ) hF_cont hF_anti hf h_tendsto


/-- **Dini's theorem** If `F n` is a monotone decreasing collection of continuous functions on a
compact space converging pointwise to a continuous function `f`, then `F n` converges uniformly
to `f`. -/
lemma tendstoUniformly_of_forall_tendsto [CompactSpace α] (hF_cont : ∀ i, Continuous (F i))
    (hF_anti : Antitone F) (hf : Continuous f) (h_tendsto : ∀ x, Tendsto (F · x) atTop (𝓝 (f x))) :
    TendstoUniformly F f atTop :=
  Monotone.tendstoUniformly_of_forall_tendsto (G := Gᵒᵈ) hF_cont hF_anti hf h_tendsto


/-- **Dini's theorem** If `F n` is a monotone decreasing collection of continuous functions on a
compact set `s` converging pointwise to a continuous `f`, then `F n` converges uniformly to `f`. -/
lemma tendstoUniformlyOn_of_forall_tendsto {s : Set α} (hs : IsCompact s)
    (hF_cont : ∀ i, ContinuousOn (F i) s) (hF_anti : ∀ x ∈ s, Antitone (F · x))
    (hf : ContinuousOn f s) (h_tendsto : ∀ x ∈ s, Tendsto (F · x) atTop (𝓝 (f x))) :
    TendstoUniformlyOn F f atTop s :=
  Monotone.tendstoUniformlyOn_of_forall_tendsto (G := Gᵒᵈ) hs hF_cont hF_anti hf h_tendsto


/-- **Dini's theorem** If `F n` is a monotone increasing collection of continuous functions
converging pointwise to a continuous function `f`, then `F n` converges to `f` in the
compact-open topology. -/
lemma tendsto_of_monotone_of_pointwise (hF_mono : Monotone F)
    (h_tendsto : ∀ x, Tendsto (F · x) atTop (𝓝 (f x))) :
    Tendsto F atTop (𝓝 f) :=
  tendsto_of_tendstoLocallyUniformly <|
    hF_mono.tendstoLocallyUniformly_of_forall_tendsto (F · |>.continuous) f.continuous h_tendsto


/-- **Dini's theorem** If `F n` is a monotone decreasing collection of continuous functions
converging pointwise to a continuous function `f`, then `F n` converges to `f` in the
compact-open topology. -/
lemma tendsto_of_antitone_of_pointwise (hF_anti : Antitone F)
    (h_tendsto : ∀ x, Tendsto (F · x) atTop (𝓝 (f x))) :
    Tendsto F atTop (𝓝 f) :=
  tendsto_of_monotone_of_pointwise (G := Gᵒᵈ) hF_anti h_tendsto


