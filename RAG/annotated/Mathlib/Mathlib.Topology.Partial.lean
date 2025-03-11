theorem rtendsto_nhds {r : Rel Y X} {l : Filter Y} {x : X} :
    RTendsto r l (𝓝 x) ↔ ∀ s, IsOpen s → x ∈ s → r.core s ∈ l :=
  all_mem_nhds_filter _ _ (fun _s _t => id) _


theorem rtendsto'_nhds {r : Rel Y X} {l : Filter Y} {x : X} :
    RTendsto' r l (𝓝 x) ↔ ∀ s, IsOpen s → x ∈ s → r.preimage s ∈ l := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    r : Rel Y X
    l : Filter Y
    x : X
    ⊢ Iff (Filter.RTendsto' r l (nhds x)) (∀ (s : Set X), IsOpen s → Membership.me …
  -/
  rw [rtendsto'_def]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    r : Rel Y X
    l : Filter Y
    x : X
    ⊢ Iff (∀ (s : Set X), Membership.mem (nhds x) s → Membership.mem l (r.preimage …
  -/
  apply all_mem_nhds_filter
  /-
    case hf
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    r : Rel Y X
    l : Filter Y
    x : X
    ⊢ ∀ (s t : Set X), HasSubset.Subset s t → HasSubset.Subset (r.preimage s) (r.p …
  -/
  apply Rel.preimage_mono
  /-
    🎉 no goals
  -/


theorem ptendsto_nhds {f : Y →. X} {l : Filter Y} {x : X} :
    PTendsto f l (𝓝 x) ↔ ∀ s, IsOpen s → x ∈ s → f.core s ∈ l :=
  rtendsto_nhds


theorem ptendsto'_nhds {f : Y →. X} {l : Filter Y} {x : X} :
    PTendsto' f l (𝓝 x) ↔ ∀ s, IsOpen s → x ∈ s → f.preimage s ∈ l :=
  rtendsto'_nhds


/-- Continuity of a partial function -/
def PContinuous (f : X →. Y) :=
  ∀ s, IsOpen s → IsOpen (f.preimage s)


theorem open_dom_of_pcontinuous {f : X →. Y} (h : PContinuous f) : IsOpen f.Dom := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : PFun X Y
    h : PContinuous f
    ⊢ IsOpen f.Dom
  -/
  rw [← PFun.preimage_univ]; exact h _ isOpen_univ
                             /-
                               🎉 no goals
                             -/


theorem pcontinuous_iff' {f : X →. Y} :
    PContinuous f ↔ ∀ {x y} (_ : y ∈ f x), PTendsto' f (𝓝 x) (𝓝 y) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : PFun X Y
    ⊢ Iff (PContinuous f) (∀ {x : X} {y : Y}, Membership.mem (f x) y → Filter.PTen …
  -/
  constructor
    /-
      case mp
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : PFun X Y
      ⊢ PContinuous f → ∀ {x : X} {y : Y}, Membership.mem (f x) y → Filter.PTendsto' …
    -/
  · intro h x y h'
    /-
      case mp
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : PFun X Y
      h : PContinuous f
      x : X
      y : Y
      h' : Membership.mem (f x) y
      ⊢ Filter.PTendsto' f (nhds x) (nhds y)
    -/
    simp only [ptendsto'_def, mem_nhds_iff]
    /-
      case mp
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : PFun X Y
      h : PContinuous f
      x : X
      y : Y
      h' : Membership.mem (f x) y
      ⊢ ∀ (s : Set Y), (Exists fun t => And (HasSubset.Subset t s) (And (IsOpen t) ( …
    -/
    rintro s ⟨t, tsubs, opent, yt⟩
    /-
      case mp.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : PFun X Y
      h : PContinuous f
      x : X
      y : Y
      h' : Membership.mem (f x) y
      s t : Set Y
      tsubs : HasSubset.Subset t s
      opent : IsOpen t
      yt : Membership.mem t y
      ⊢ Exists fun t => And (HasSubset.Subset t (f.preimage s)) (And (IsOpen t) (Mem …
    -/
    exact ⟨f.preimage t, PFun.preimage_mono _ tsubs, h _ opent, ⟨y, yt, h'⟩⟩
    /-
      🎉 no goals
    -/
  /-
    case mpr
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : PFun X Y
    ⊢ (∀ {x : X} {y : Y}, Membership.mem (f x) y → Filter.PTendsto' f (nhds x) (nh …
  -/
  intro hf s os
  /-
    case mpr
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : PFun X Y
    hf : ∀ {x : X} {y : Y}, Membership.mem (f x) y → Filter.PTendsto' f (nhds x) ( …
    s : Set Y
    os : IsOpen s
    ⊢ IsOpen (f.preimage s)
  -/
  rw [isOpen_iff_nhds]
  /-
    case mpr
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : PFun X Y
    hf : ∀ {x : X} {y : Y}, Membership.mem (f x) y → Filter.PTendsto' f (nhds x) ( …
    s : Set Y
    os : IsOpen s
    ⊢ ∀ (x : X), Membership.mem (f.preimage s) x → LE.le (nhds x) (Filter.principa …
  -/
  rintro x ⟨y, ys, fxy⟩ t
  /-
    case mpr.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : PFun X Y
    hf : ∀ {x : X} {y : Y}, Membership.mem (f x) y → Filter.PTendsto' f (nhds x) ( …
    s : Set Y
    os : IsOpen s
    x : X
    y : Y
    ys : Membership.mem s y
    fxy : Membership.mem (f x) y
    t : Set X
    ⊢ Membership.mem (Filter.principal (f.preimage s)) t → Membership.mem (nhds x) t
  -/
  rw [mem_principal]
  /-
    case mpr.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : PFun X Y
    hf : ∀ {x : X} {y : Y}, Membership.mem (f x) y → Filter.PTendsto' f (nhds x) ( …
    s : Set Y
    os : IsOpen s
    x : X
    y : Y
    ys : Membership.mem s y
    fxy : Membership.mem (f x) y
    t : Set X
    ⊢ HasSubset.Subset (f.preimage s) t → Membership.mem (nhds x) t
  -/
  intro (h : f.preimage s ⊆ t)
  /-
    case mpr.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : PFun X Y
    hf : ∀ {x : X} {y : Y}, Membership.mem (f x) y → Filter.PTendsto' f (nhds x) ( …
    s : Set Y
    os : IsOpen s
    x : X
    y : Y
    ys : Membership.mem s y
    fxy : Membership.mem (f x) y
    t : Set X
    h : HasSubset.Subset (f.preimage s) t
    ⊢ Membership.mem (nhds x) t
  -/
  apply mem_of_superset _ h
  have h' : ∀ s ∈ 𝓝 y, f.preimage s ∈ 𝓝 x := by
    intro s hs
    have : PTendsto' f (𝓝 x) (𝓝 y) := hf fxy
    rw [ptendsto'_def] at this
    exact this s hs
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : PFun X Y
    hf : ∀ {x : X} {y : Y}, Membership.mem (f x) y → Filter.PTendsto' f (nhds x) ( …
    s : Set Y
    os : IsOpen s
    x : X
    y : Y
    ys : Membership.mem s y
    fxy : Membership.mem (f x) y
    t : Set X
    h : HasSubset.Subset (f.preimage s) t
    h' : ∀ (s : Set Y), Membership.mem (nhds y) s → Membership.mem (nhds x) (f.pre …
    ⊢ Membership.mem (nhds x) (f.preimage s)
  -/
  show f.preimage s ∈ 𝓝 x
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : PFun X Y
    hf : ∀ {x : X} {y : Y}, Membership.mem (f x) y → Filter.PTendsto' f (nhds x) ( …
    s : Set Y
    os : IsOpen s
    x : X
    y : Y
    ys : Membership.mem s y
    fxy : Membership.mem (f x) y
    t : Set X
    h : HasSubset.Subset (f.preimage s) t
    h' : ∀ (s : Set Y), Membership.mem (nhds y) s → Membership.mem (nhds x) (f.pre …
    ⊢ Membership.mem (nhds x) (f.preimage s)
  -/
  apply h'
  /-
    case a
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : PFun X Y
    hf : ∀ {x : X} {y : Y}, Membership.mem (f x) y → Filter.PTendsto' f (nhds x) ( …
    s : Set Y
    os : IsOpen s
    x : X
    y : Y
    ys : Membership.mem s y
    fxy : Membership.mem (f x) y
    t : Set X
    h : HasSubset.Subset (f.preimage s) t
    h' : ∀ (s : Set Y), Membership.mem (nhds y) s → Membership.mem (nhds x) (f.pre …
    ⊢ Membership.mem (nhds y) s
  -/
  rw [mem_nhds_iff]
  /-
    case a
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : PFun X Y
    hf : ∀ {x : X} {y : Y}, Membership.mem (f x) y → Filter.PTendsto' f (nhds x) ( …
    s : Set Y
    os : IsOpen s
    x : X
    y : Y
    ys : Membership.mem s y
    fxy : Membership.mem (f x) y
    t : Set X
    h : HasSubset.Subset (f.preimage s) t
    h' : ∀ (s : Set Y), Membership.mem (nhds y) s → Membership.mem (nhds x) (f.pre …
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And (IsOpen t) (Membership.mem t …
  -/
  exact ⟨s, Set.Subset.refl _, os, ys⟩
  /-
    🎉 no goals
  -/


theorem continuousWithinAt_iff_ptendsto_res (f : X → Y) {x : X} {s : Set X} :
    ContinuousWithinAt f s x ↔ PTendsto (PFun.res f s) (𝓝 x) (𝓝 (f x)) :=
  tendsto_iff_ptendsto _ _ _ _

