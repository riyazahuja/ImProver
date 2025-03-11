variable (I) in
/-- Given a model with corners `(E, H)`, we define the pregroupoid of analytic transformations of
`H` as the maps that are `AnalyticOn` when read in `E` through `I`.  Using `AnalyticOn`
rather than `AnalyticOnNhd` gives us meaningful definitions at boundary points. -/
def analyticPregroupoid : Pregroupoid H where
  property f s := AnalyticOn 𝕜 (I ∘ f ∘ I.symm) (I.symm ⁻¹' s ∩ range I)
  comp {f g u v} hf hg _ _ _ := by
    /-
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u v : Set H
      hf : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
      hg : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
      x✝² : IsOpen u
      x✝¹ : IsOpen v
      x✝ : IsOpen (Inter.inter u (Set.preimage f v))
      ⊢ (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) (Int …
    -/
    have : I ∘ (g ∘ f) ∘ I.symm = (I ∘ g ∘ I.symm) ∘ I ∘ f ∘ I.symm := by ext x; simp
    /-
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u v : Set H
      hf : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
      hg : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
      x✝² : IsOpen u
      x✝¹ : IsOpen v
      x✝ : IsOpen (Inter.inter u (Set.preimage f v))
      this : Eq (Function.comp (↑I) (Function.comp (Function.comp g f) ↑I.symm)) (Fu …
      ⊢ (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) (Int …
    -/
    simp only [this]
    /-
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u v : Set H
      hf : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
      hg : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
      x✝² : IsOpen u
      x✝¹ : IsOpen v
      x✝ : IsOpen (Inter.inter u (Set.preimage f v))
      this : Eq (Function.comp (↑I) (Function.comp (Function.comp g f) ↑I.symm)) (Fu …
      ⊢ AnalyticOn 𝕜 (Function.comp (Function.comp (↑I) (Function.comp g ↑I.symm)) ( …
    -/
    apply hg.comp
      /-
        case hg
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        H : Type u_3
        inst✝¹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝ : TopologicalSpace M
        f g : H → H
        u v : Set H
        hf : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
        hg : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
        x✝² : IsOpen u
        x✝¹ : IsOpen v
        x✝ : IsOpen (Inter.inter u (Set.preimage f v))
        this : Eq (Function.comp (↑I) (Function.comp (Function.comp g f) ↑I.symm)) (Fu …
        ⊢ AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) (Inter.inter (Se …
      -/
    · exact hf.mono fun _ ⟨hx1, hx2⟩ ↦ ⟨hx1.1, hx2⟩
      /-
        🎉 no goals
      -/
      /-
        case h
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        H : Type u_3
        inst✝¹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝ : TopologicalSpace M
        f g : H → H
        u v : Set H
        hf : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
        hg : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
        x✝² : IsOpen u
        x✝¹ : IsOpen v
        x✝ : IsOpen (Inter.inter u (Set.preimage f v))
        this : Eq (Function.comp (↑I) (Function.comp (Function.comp g f) ↑I.symm)) (Fu …
        ⊢ Set.MapsTo (Function.comp (↑I) (Function.comp f ↑I.symm)) (Inter.inter (Set. …
      -/
    · rintro x ⟨hx1, _⟩
      /-
        case h.intro
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        H : Type u_3
        inst✝¹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝ : TopologicalSpace M
        f g : H → H
        u v : Set H
        hf : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
        hg : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
        x✝² : IsOpen u
        x✝¹ : IsOpen v
        x✝ : IsOpen (Inter.inter u (Set.preimage f v))
        this : Eq (Function.comp (↑I) (Function.comp (Function.comp g f) ↑I.symm)) (Fu …
        x : E
        hx1 : Membership.mem (Set.preimage (↑I.symm) (Inter.inter u (Set.preimage f v) …
        right✝ : Membership.mem (Set.range ↑I) x
        ⊢ Membership.mem (Inter.inter (Set.preimage (↑I.symm) v) (Set.range ↑I)) (Func …
      -/
      simpa only [mfld_simps] using hx1.2
      /-
        🎉 no goals
      -/
  id_mem := by
    /-
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      ⊢ (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) (Int …
    -/
    apply analyticOn_id.congr
    /-
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      ⊢ Set.EqOn (Function.comp (↑I) (Function.comp id ↑I.symm)) (fun x => x) (Inter …
    -/
    rintro x ⟨_, hx2⟩
    /-
      case intro
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      x : E
      left✝ : Membership.mem (Set.preimage (↑I.symm) Set.univ) x
      hx2 : Membership.mem (Set.range ↑I) x
      ⊢ Eq (Function.comp (↑I) (Function.comp id ↑I.symm) x) ((fun x => x) x)
    -/
    obtain ⟨y, hy⟩ := mem_range.1 hx2
    /-
      case intro.intro
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      x : E
      left✝ : Membership.mem (Set.preimage (↑I.symm) Set.univ) x
      hx2 : Membership.mem (Set.range ↑I) x
      y : H
      hy : Eq (↑I y) x
      ⊢ Eq (Function.comp (↑I) (Function.comp id ↑I.symm) x) ((fun x => x) x)
    -/
    simp only [mfld_simps, ← hy]
    /-
      🎉 no goals
    -/
  locality {f u} _ H := by
    /-
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H✝ : Type u_3
      inst✝¹ : TopologicalSpace H✝
      I : ModelWithCorners 𝕜 E H✝
      M : Type u_4
      inst✝ : TopologicalSpace M
      f : H✝ → H✝
      u : Set H✝
      x✝ : IsOpen u
      H : ∀ (x : H✝), Membership.mem u x → Exists fun v => And (IsOpen v) (And (Memb …
      ⊢ (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) (Int …
    -/
    apply analyticOn_of_locally_analyticOn
    /-
      case h
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H✝ : Type u_3
      inst✝¹ : TopologicalSpace H✝
      I : ModelWithCorners 𝕜 E H✝
      M : Type u_4
      inst✝ : TopologicalSpace M
      f : H✝ → H✝
      u : Set H✝
      x✝ : IsOpen u
      H : ∀ (x : H✝), Membership.mem u x → Exists fun v => And (IsOpen v) (And (Memb …
      ⊢ ∀ (x : E), Membership.mem (Inter.inter (Set.preimage (↑I.symm) u) (Set.range …
    -/
    rintro y ⟨hy1, hy2⟩
    /-
      case h.intro
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H✝ : Type u_3
      inst✝¹ : TopologicalSpace H✝
      I : ModelWithCorners 𝕜 E H✝
      M : Type u_4
      inst✝ : TopologicalSpace M
      f : H✝ → H✝
      u : Set H✝
      x✝ : IsOpen u
      H : ∀ (x : H✝), Membership.mem u x → Exists fun v => And (IsOpen v) (And (Memb …
      y : E
      hy1 : Membership.mem (Set.preimage (↑I.symm) u) y
      hy2 : Membership.mem (Set.range ↑I) y
      ⊢ Exists fun u_1 => And (IsOpen u_1) (And (Membership.mem u_1 y) (AnalyticOn 𝕜 …
    -/
    obtain ⟨x, hx⟩ := mem_range.1 hy2
    /-
      case h.intro.intro
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H✝ : Type u_3
      inst✝¹ : TopologicalSpace H✝
      I : ModelWithCorners 𝕜 E H✝
      M : Type u_4
      inst✝ : TopologicalSpace M
      f : H✝ → H✝
      u : Set H✝
      x✝ : IsOpen u
      H : ∀ (x : H✝), Membership.mem u x → Exists fun v => And (IsOpen v) (And (Memb …
      y : E
      hy1 : Membership.mem (Set.preimage (↑I.symm) u) y
      hy2 : Membership.mem (Set.range ↑I) y
      x : H✝
      hx : Eq (↑I x) y
      ⊢ Exists fun u_1 => And (IsOpen u_1) (And (Membership.mem u_1 y) (AnalyticOn 𝕜 …
    -/
    simp only [mfld_simps, ← hx] at hy1 ⊢
    /-
      case h.intro.intro
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H✝ : Type u_3
      inst✝¹ : TopologicalSpace H✝
      I : ModelWithCorners 𝕜 E H✝
      M : Type u_4
      inst✝ : TopologicalSpace M
      f : H✝ → H✝
      u : Set H✝
      x✝ : IsOpen u
      H : ∀ (x : H✝), Membership.mem u x → Exists fun v => And (IsOpen v) (And (Memb …
      y : E
      hy2 : Membership.mem (Set.range ↑I) y
      x : H✝
      hx : Eq (↑I x) y
      hy1 : Membership.mem u x
      ⊢ Exists fun u_1 => And (IsOpen u_1) (And (Membership.mem u_1 (↑I x)) (Analyti …
    -/
    obtain ⟨v, v_open, xv, hv⟩ := H x hy1
    have : I.symm ⁻¹' (u ∩ v) ∩ range I = I.symm ⁻¹' u ∩ range I ∩ I.symm ⁻¹' v := by
      rw [preimage_inter, inter_assoc, inter_assoc, inter_comm _ (range I)]
    /-
      case h.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H✝ : Type u_3
      inst✝¹ : TopologicalSpace H✝
      I : ModelWithCorners 𝕜 E H✝
      M : Type u_4
      inst✝ : TopologicalSpace M
      f : H✝ → H✝
      u : Set H✝
      x✝ : IsOpen u
      H : ∀ (x : H✝), Membership.mem u x → Exists fun v => And (IsOpen v) (And (Memb …
      y : E
      hy2 : Membership.mem (Set.range ↑I) y
      x : H✝
      hx : Eq (↑I x) y
      hy1 : Membership.mem u x
      v : Set H✝
      v_open : IsOpen v
      xv : Membership.mem v x
      hv : AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) (Inter.inter  …
      this : Eq (Inter.inter (Set.preimage (↑I.symm) (Inter.inter u v)) (Set.range ↑ …
      ⊢ Exists fun u_1 => And (IsOpen u_1) (And (Membership.mem u_1 (↑I x)) (Analyti …
    -/
    exact ⟨I.symm ⁻¹' v, v_open.preimage I.continuous_symm, by simpa, this ▸ hv⟩
    /-
      🎉 no goals
    -/
  congr {f g u} _ fg hf := by
    /-
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u : Set H
      x✝ : IsOpen u
      fg : ∀ (x : H), Membership.mem u x → Eq (g x) (f x)
      hf : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
      ⊢ (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) (Int …
    -/
    apply hf.congr
    /-
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u : Set H
      x✝ : IsOpen u
      fg : ∀ (x : H), Membership.mem u x → Eq (g x) (f x)
      hf : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
      ⊢ Set.EqOn (Function.comp (↑I) (Function.comp g ↑I.symm)) (Function.comp (↑I)  …
    -/
    rintro y ⟨hy1, hy2⟩
    /-
      case intro
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u : Set H
      x✝ : IsOpen u
      fg : ∀ (x : H), Membership.mem u x → Eq (g x) (f x)
      hf : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
      y : E
      hy1 : Membership.mem (Set.preimage (↑I.symm) u) y
      hy2 : Membership.mem (Set.range ↑I) y
      ⊢ Eq (Function.comp (↑I) (Function.comp g ↑I.symm) y) (Function.comp (↑I) (Fun …
    -/
    obtain ⟨x, hx⟩ := mem_range.1 hy2
    /-
      case intro.intro
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u : Set H
      x✝ : IsOpen u
      fg : ∀ (x : H), Membership.mem u x → Eq (g x) (f x)
      hf : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
      y : E
      hy1 : Membership.mem (Set.preimage (↑I.symm) u) y
      hy2 : Membership.mem (Set.range ↑I) y
      x : H
      hx : Eq (↑I x) y
      ⊢ Eq (Function.comp (↑I) (Function.comp g ↑I.symm) y) (Function.comp (↑I) (Fun …
    -/
    simp only [mfld_simps, ← hx] at hy1 ⊢
    /-
      case intro.intro
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u : Set H
      x✝ : IsOpen u
      fg : ∀ (x : H), Membership.mem u x → Eq (g x) (f x)
      hf : (fun f s => AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp f ↑I.symm)) ( …
      y : E
      hy2 : Membership.mem (Set.range ↑I) y
      x : H
      hx : Eq (↑I x) y
      hy1 : Membership.mem u x
      ⊢ Eq (↑I (g x)) (↑I (f x))
    -/
    rw [fg _ hy1]
    /-
      🎉 no goals
    -/


variable (I) in
/-- Given a model with corners `(E, H)`, we define the groupoid of analytic transformations of
`H` as the maps that are `AnalyticOn` when read in `E` through `I`.  Using `AnalyticOn`
rather than `AnalyticOnNhd` gives us meaningful definitions at boundary points. -/
def analyticGroupoid : StructureGroupoid H :=
  (analyticPregroupoid I).groupoid


/-- An identity partial homeomorphism belongs to the analytic groupoid. -/
theorem ofSet_mem_analyticGroupoid {s : Set H} (hs : IsOpen s) :
    PartialHomeomorph.ofSet s hs ∈ analyticGroupoid I := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    s : Set H
    hs : IsOpen s
    ⊢ Membership.mem (analyticGroupoid I) (PartialHomeomorph.ofSet s hs)
  -/
  rw [analyticGroupoid, mem_groupoid_of_pregroupoid]
  suffices h : AnalyticOn 𝕜 (I ∘ I.symm) (I.symm ⁻¹' s ∩ range I) by
    simp [h, analyticPregroupoid]
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    s : Set H
    hs : IsOpen s
    ⊢ AnalyticOn 𝕜 (Function.comp ↑I ↑I.symm) (Inter.inter (Set.preimage (↑I.symm) …
  -/
  have hi : AnalyticOn 𝕜 id (univ : Set E) := analyticOn_id
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    s : Set H
    hs : IsOpen s
    hi : AnalyticOn 𝕜 id Set.univ
    ⊢ AnalyticOn 𝕜 (Function.comp ↑I ↑I.symm) (Inter.inter (Set.preimage (↑I.symm) …
  -/
  exact (hi.mono (subset_univ _)).congr (fun x hx ↦ I.right_inv hx.2)
  /-
    🎉 no goals
  -/


/-- The composition of a partial homeomorphism from `H` to `M` and its inverse belongs to
the analytic groupoid. -/
theorem symm_trans_mem_analyticGroupoid (e : PartialHomeomorph M H) :
    e.symm.trans e ∈ analyticGroupoid I :=
  haveI : e.symm.trans e ≈ PartialHomeomorph.ofSet e.target e.open_target :=
    PartialHomeomorph.symm_trans_self _
  StructureGroupoid.mem_of_eqOnSource _ (ofSet_mem_analyticGroupoid e.open_target) this


/-- The analytic groupoid is closed under restriction. -/
instance : ClosedUnderRestriction (analyticGroupoid I) :=
  (closedUnderRestriction_iff_id_le _).mpr
    (by
      /-
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        H : Type u_3
        inst✝¹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝ : TopologicalSpace M
        ⊢ LE.le idRestrGroupoid (analyticGroupoid I)
      -/
      rw [StructureGroupoid.le_iff]
      /-
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        H : Type u_3
        inst✝¹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝ : TopologicalSpace M
        ⊢ ∀ (e : PartialHomeomorph H H), Membership.mem idRestrGroupoid e → Membership …
      -/
      rintro e ⟨s, hs, hes⟩
      /-
        case intro.intro
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        H : Type u_3
        inst✝¹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝ : TopologicalSpace M
        e : PartialHomeomorph H H
        s : Set H
        hs : IsOpen s
        hes : HasEquiv.Equiv e (PartialHomeomorph.ofSet s hs)
        ⊢ Membership.mem (analyticGroupoid I) e
      -/
      exact (analyticGroupoid I).mem_of_eqOnSource' _ _ (ofSet_mem_analyticGroupoid hs) hes)
      /-
        🎉 no goals
      -/


/-- `f ∈ analyticGroupoid` iff it and its inverse are analytic within `range I`. -/
lemma mem_analyticGroupoid {I : ModelWithCorners 𝕜 E H} {f : PartialHomeomorph H H} :
    f ∈ analyticGroupoid I ↔
      AnalyticOn 𝕜 (I ∘ f ∘ I.symm) (I.symm ⁻¹' f.source ∩ range I) ∧
      AnalyticOn 𝕜 (I ∘ f.symm ∘ I.symm) (I.symm ⁻¹' f.target ∩ range I) := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    f : PartialHomeomorph H H
    ⊢ Iff (Membership.mem (analyticGroupoid I) f) (And (AnalyticOn 𝕜 (Function.com …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The analytic groupoid on a boundaryless charted space modeled on a complete vector space
consists of the partial homeomorphisms which are analytic and have analytic inverse. -/
theorem mem_analyticGroupoid_of_boundaryless [I.Boundaryless] (e : PartialHomeomorph H H) :
    e ∈ analyticGroupoid I ↔ AnalyticOnNhd 𝕜 (I ∘ e ∘ I.symm) (I '' e.source) ∧
      AnalyticOnNhd 𝕜 (I ∘ e.symm ∘ I.symm) (I '' e.target) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    inst✝ : I.Boundaryless
    e : PartialHomeomorph H H
    ⊢ Iff (Membership.mem (analyticGroupoid I) e) (And (AnalyticOnNhd 𝕜 (Function. …
  -/
  simp only [mem_analyticGroupoid, I.range_eq_univ, inter_univ, I.image_eq]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    inst✝ : I.Boundaryless
    e : PartialHomeomorph H H
    ⊢ Iff (And (AnalyticOn 𝕜 (Function.comp (↑I) (Function.comp ↑e ↑I.symm)) (Set. …
  -/
  rw [IsOpen.analyticOn_iff_analyticOnNhd, IsOpen.analyticOn_iff_analyticOnNhd]
    /-
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      inst✝ : I.Boundaryless
      e : PartialHomeomorph H H
      ⊢ IsOpen (Set.preimage (↑I.symm) e.target)
    -/
  · exact I.continuous_symm.isOpen_preimage _ e.open_target
    /-
      🎉 no goals
    -/
    /-
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      inst✝ : I.Boundaryless
      e : PartialHomeomorph H H
      ⊢ IsOpen (Set.preimage (↑I.symm) e.source)
    -/
  · exact I.continuous_symm.isOpen_preimage _ e.open_source
    /-
      🎉 no goals
    -/


/-- `analyticGroupoid` is closed under products -/
theorem analyticGroupoid_prod {E A : Type} [NormedAddCommGroup E] [NormedSpace 𝕜 E]
    [TopologicalSpace A] {F B : Type} [NormedAddCommGroup F] [NormedSpace 𝕜 F]
    [TopologicalSpace B] {I : ModelWithCorners 𝕜 E A} {J : ModelWithCorners 𝕜 F B}
    {f : PartialHomeomorph A A} {g : PartialHomeomorph B B}
    (fa : f ∈ analyticGroupoid I) (ga : g ∈ analyticGroupoid J) :
    f.prod g ∈ analyticGroupoid (I.prod J) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E A : Type
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : TopologicalSpace A
    F B : Type
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : TopologicalSpace B
    I : ModelWithCorners 𝕜 E A
    J : ModelWithCorners 𝕜 F B
    f : PartialHomeomorph A A
    g : PartialHomeomorph B B
    fa : Membership.mem (analyticGroupoid I) f
    ga : Membership.mem (analyticGroupoid J) g
    ⊢ Membership.mem (analyticGroupoid (I.prod J)) (f.prod g)
  -/
  have pe : range (I.prod J) = (range I).prod (range J) := I.range_prod
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E A : Type
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : TopologicalSpace A
    F B : Type
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : TopologicalSpace B
    I : ModelWithCorners 𝕜 E A
    J : ModelWithCorners 𝕜 F B
    f : PartialHomeomorph A A
    g : PartialHomeomorph B B
    fa : Membership.mem (analyticGroupoid I) f
    ga : Membership.mem (analyticGroupoid J) g
    pe : Eq (Set.range ↑(I.prod J)) ((Set.range ↑I).prod (Set.range ↑J))
    ⊢ Membership.mem (analyticGroupoid (I.prod J)) (f.prod g)
  -/
  simp only [mem_analyticGroupoid, Function.comp, image_subset_iff] at fa ga ⊢
  exact ⟨AnalyticOn.prod
      (fa.1.comp analyticOn_fst fun _ m ↦ ⟨m.1.1, (pe ▸ m.2).1⟩)
      (ga.1.comp analyticOn_snd fun _ m ↦ ⟨m.1.2, (pe ▸ m.2).2⟩),
    AnalyticOn.prod
      (fa.2.comp analyticOn_fst fun _ m ↦ ⟨m.1.1, (pe ▸ m.2).1⟩)
      (ga.2.comp analyticOn_snd fun _ m ↦ ⟨m.1.2, (pe ▸ m.2).2⟩)⟩


/-- An analytic manifold w.r.t. a model `I : ModelWithCorners 𝕜 E H` is a charted space over `H`
s.t. all extended chart conversion maps are analytic. -/
class AnalyticManifold (I : ModelWithCorners 𝕜 E H) (M : Type*) [TopologicalSpace M]
  [ChartedSpace H M] extends HasGroupoid M (analyticGroupoid I) : Prop


/-- Normed spaces are analytic manifolds over themselves. -/
instance AnalyticManifold.self : AnalyticManifold 𝓘(𝕜, E) E where


/-- `M × N` is an analytic manifold if `M` and `N` are -/
instance AnalyticManifold.prod {E A : Type} [NormedAddCommGroup E] [NormedSpace 𝕜 E]
    [TopologicalSpace A] {F B : Type} [NormedAddCommGroup F] [NormedSpace 𝕜 F]
    [TopologicalSpace B] {I : ModelWithCorners 𝕜 E A} {J : ModelWithCorners 𝕜 F B}
    {M : Type} [TopologicalSpace M] [ChartedSpace A M] [m : AnalyticManifold I M]
    {N : Type} [TopologicalSpace N] [ChartedSpace B N] [n : AnalyticManifold J N] :
    AnalyticManifold (I.prod J) (M × N) where
  compatible := by
    /-
      𝕜 : Type u_1
      inst✝¹⁴ : NontriviallyNormedField 𝕜
      E✝ : Type u_2
      inst✝¹³ : NormedAddCommGroup E✝
      inst✝¹² : NormedSpace 𝕜 E✝
      H : Type u_3
      inst✝¹¹ : TopologicalSpace H
      I✝ : ModelWithCorners 𝕜 E✝ H
      M✝ : Type u_4
      inst✝¹⁰ : TopologicalSpace M✝
      E A : Type
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace 𝕜 E
      inst✝⁷ : TopologicalSpace A
      F B : Type
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace 𝕜 F
      inst✝⁴ : TopologicalSpace B
      I : ModelWithCorners 𝕜 E A
      J : ModelWithCorners 𝕜 F B
      M : Type
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace A M
      m : AnalyticManifold I M
      N : Type
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace B N
      n : AnalyticManifold J N
      ⊢ ∀ {e e' : PartialHomeomorph (Prod M N) (ModelProd A B)}, Membership.mem (atl …
    -/
    intro f g ⟨f1, f2, hf1, hf2, fe⟩ ⟨g1, g2, hg1, hg2, ge⟩
    /-
      𝕜 : Type u_1
      inst✝¹⁴ : NontriviallyNormedField 𝕜
      E✝ : Type u_2
      inst✝¹³ : NormedAddCommGroup E✝
      inst✝¹² : NormedSpace 𝕜 E✝
      H : Type u_3
      inst✝¹¹ : TopologicalSpace H
      I✝ : ModelWithCorners 𝕜 E✝ H
      M✝ : Type u_4
      inst✝¹⁰ : TopologicalSpace M✝
      E A : Type
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace 𝕜 E
      inst✝⁷ : TopologicalSpace A
      F B : Type
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace 𝕜 F
      inst✝⁴ : TopologicalSpace B
      I : ModelWithCorners 𝕜 E A
      J : ModelWithCorners 𝕜 F B
      M : Type
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace A M
      m : AnalyticManifold I M
      N : Type
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace B N
      n : AnalyticManifold J N
      f g : PartialHomeomorph (Prod M N) (ModelProd A B)
      f1 : PartialHomeomorph M A
      f2 : Membership.mem (atlas A M) f1
      hf1 : PartialHomeomorph N B
      hf2 : Membership.mem (atlas B N) hf1
      fe : Eq (f1.prod hf1) f
      g1 : PartialHomeomorph M A
      g2 : Membership.mem (atlas A M) g1
      hg1 : PartialHomeomorph N B
      hg2 : Membership.mem (atlas B N) hg1
      ge : Eq (g1.prod hg1) g
      ⊢ Membership.mem (analyticGroupoid (I.prod J)) (f.symm.trans g)
    -/
    rw [← fe, ← ge, PartialHomeomorph.prod_symm, PartialHomeomorph.prod_trans]
    exact analyticGroupoid_prod (m.toHasGroupoid.compatible f2 g2)
      (n.toHasGroupoid.compatible hf2 hg2)


/-- Analytic manifolds are smooth manifolds. -/
instance AnalyticManifold.smoothManifoldWithCorners [ChartedSpace H M]
    [cm : AnalyticManifold I M] :
    SmoothManifoldWithCorners I M where
  compatible hf hg := ⟨(cm.compatible hf hg).1.contDiffOn I.uniqueDiffOn_preimage_source,
    (cm.compatible hg hf).1.contDiffOn I.uniqueDiffOn_preimage_source⟩



